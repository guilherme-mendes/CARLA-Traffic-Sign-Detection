from PIL import Image
import cv2
import numpy as np
import glob
import pytesseract as ocr

escale = 5
signs = {'T': 'Stop',
		 'P': 'Stop',
		 '3': 30,
		 '6': 60,
		 '9': 90}

def image_to_string(img):

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)

	image = im_pil
	width,height= image.size
	#image.resize((width*escale,height*escale))
	result = ocr.image_to_string(image)

	for key in signs.keys():
		if key in result:
			print(signs[key])
			return signs[key]
			break

	# print(result)
