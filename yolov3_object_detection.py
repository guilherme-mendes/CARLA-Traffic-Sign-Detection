#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import cv2
import time
import numpy as np
import argparse

import tensorflow_yolov3.carla.utils as utils
from traffic_sign import Sign

import tensorflow as tf
from PIL import Image


import glob
import os
import sys


try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = None
VIEW_HEIGHT = None
MAX_FPS = None
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

os.system('cls' if os.name == 'nt' else 'clear')

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.raw_image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        #Spawn near to a Traffic sign
        #location = carla.Transform(carla.Location(x=-246.670059, y=-3.667096, z=-0.009936), carla.Rotation(pitch=0.018756, yaw=174.198608, roll=-0.063385)) 
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        #First person view transform settings
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            # self.camera.destroy()
            # self.car.destroy()
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))




    def game_loop(self, num_classes, input_size, graph, return_tensors):
        """
        Main program loop.
        """        
        try:
            pygame.init()
            
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            #self.world = self.client.load_world('Town05')

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            # CLEAR TERMINAL
            os.system('cls' if os.name == 'nt' else 'clear')
            sign = Sign()
            
            aux = ''

            with tf.Session(graph=graph) as sess:
                while True:
                    self.world.tick()
    
                    self.capture = True
                    pygame_clock.tick_busy_loop(20)
    
                    self.render(self.display)
                    frame = self.raw_image.copy()
                    self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                    frame_size = self.raw_image.shape[:2]
                    image_data = utils.image_preporcess(np.copy(self.raw_image), [input_size, input_size])
                    image_data = image_data[np.newaxis, ...]
                    # PREDITCTED MODELS:
                    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                        [return_tensors[1], return_tensors[2], return_tensors[3]],
                                feed_dict={ return_tensors[0]: image_data})

                    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
                    
                    bboxes =  utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
                    bboxes =  utils.nms(bboxes, 0.45, method='nms')

                    bboxes = sign.filter_traffic_sign(bboxes)
                    spt = str(sign.process_traffic_sign(frame, bboxes))
                

                    if(spt != 'None'):
                        aux = spt
                    basicfont = pygame.font.SysFont(None, 60)
                    text_control = basicfont.render(aux, True, (0,0,0))
                    textrec = text_control.get_rect()
                    textrec.top = self.display.get_rect().top
                    textrec.bottomleft = self.display.get_rect().bottomleft
                    self.display.blit(text_control, textrec)

                    utils.draw_bounding_boxes(pygame, self.display,  self.raw_image, bboxes)
                    
                    pygame.display.flip()
    
                    pygame.event.pump()
                    if self.control(self.car):
                        return
        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        
        return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        pb_file         = "tensorflow_yolov3/yolov3_coco.pb"
        
        # video_path      = 0
        num_classes     = 80
        input_size      = 416
        graph           = tf.Graph()
        
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, pb_file)
        print("my_file:", my_file)
        
        return_tensors  =  utils.read_pb_return_tensors(graph, my_file, return_elements)
        
        client = BasicSynchronousClient()
        client.game_loop(num_classes, input_size, graph, return_tensors)

    finally:
        print('EXIT')


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-wi", "--width", type=int, default=720,
                    help="Video widht")
    ap.add_argument("-he", "--height", type=int, default=480,
                    help="Video height")
    ap.add_argument("-f", "--fps_max", type=int, default=60,
                    help="Video max fps")
    args = vars(ap.parse_args())
    
    VIEW_WIDTH = args["width"]
    VIEW_HEIGHT = args["height"]
    MAX_FPS = args["fps_max"]
    main()

