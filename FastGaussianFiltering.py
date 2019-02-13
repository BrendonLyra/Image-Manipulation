# Fast Gaussian Filter
# Brendon Lyra 2/10/19

from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt


def FastGaussian(name, stdDev):

	stdDevSqrd = stdDev**2

	# size of kernel
	size = stdDev * 6
	if size % 2 == 0:
		size += 1

	# opening image in memory, and making it an array
	img = Image.open(name)
	img_arr = np.array(img, dtype=float)

	# left hand side of 1D gaussian formula
	lhs = 1/math.sqrt(2* math.pi * stdDevSqrd)

	# both kernels to be used for convolution
	xKernel = np.ones((1, size))
	yKernel = np.ones((size, 1))

	# offset to use for x/y values in kernel
	offset = -(size//2)

	# place correct values into both the x and y gaussian kernels.
	for i in range(size):
		xKernel[0][i] = lhs*(math.exp(-((offset+i)**2)/(2*stdDevSqrd)))
		yKernel[i][0] = lhs*(math.exp(-((offset+i)**2)/(2*stdDevSqrd)))

	# running convolution on image with x kernel, and then on that new image running y kernel convolution.
	# this is allowed since gaussian is fully circular
	xConvolved = ndimage.convolve(img_arr, xKernel)
	joinedImg = ndimage.convolve(xConvolved, yKernel)

	newImage = Image.fromarray(joinedImg)
	
	return newImage


def MainFastGaussian():

	
	plt.subplot(231), plt.imshow(FastGaussian("image1.png", 3)), plt.title('img1 sigma=3'), plt.xticks([]), plt.yticks([])
	plt.subplot(232), plt.imshow(FastGaussian("image1.png", 5)), plt.title('img1 sigma=5'), plt.xticks([]), plt.yticks([])
	plt.subplot(233), plt.imshow(FastGaussian("image1.png", 10)), plt.title('img1 sigma=10'), plt.xticks([]), plt.yticks([])

	plt.subplot(234), plt.imshow(FastGaussian("image2.png", 3)), plt.title('img2 sigma=3'), plt.xticks([]), plt.yticks([])
	plt.subplot(235), plt.imshow(FastGaussian("image2.png", 5)), plt.title('img2 sigma=5'), plt.xticks([]), plt.yticks([])
	plt.subplot(236), plt.imshow(FastGaussian("image2.png", 10)), plt.title('img2 sigma=10'), plt.xticks([]), plt.yticks([])

	plt.show()

MainFastGaussian()
	
	