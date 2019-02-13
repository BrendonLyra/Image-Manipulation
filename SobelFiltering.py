# Sobel Filtering. Question 5
# Brendon Lyra 2/10/19

'''
Answering question for sobel filter:

The resulting images are black where there are no sudden changes in 
pixel values (non-edges), and it is white where there are sudden changes (edges).
It is very succeptible to noise however, and will highlight the noise as an edge, 
which is actually incorrect
'''

from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def sobelFilter(name):

	# opening image in memory, and making it an array
	img = Image.open(name)
	row, col = img.size
	img_arr = np.array(img, dtype=float)

	# kernels to use for algorithm. 
	dx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
	dy = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

	# will join both dx and dy from convolution. same size as image
	joinedImg = np.ndarray((row,col))

	# convolution for both dx and dy for image
	x_convolved = ndimage.convolve(img_arr, dx)
	y_convolved = ndimage.convolve(img_arr, dy)

	# traverse both x and y convolutions, and place the root of their squares added together
	for i in range(256):
		for j in range(256):
			joinedImg[i][j] = math.sqrt((math.pow(x_convolved[i][j],2)) + (math.pow(y_convolved[i][j], 2)))

	newImage = Image.fromarray(joinedImg)
	return newImage


# main controller function
def MainSobel():

	plt.subplot(121), plt.imshow(sobelFilter("image1.png")), plt.title('img1-Sobel'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(sobelFilter("image2.png")), plt.title('img2-Sobel'), plt.xticks([]), plt.yticks([])
	plt.show()

MainSobel()