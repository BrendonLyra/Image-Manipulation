# Gaussian Filtering. Question 3
# Brendon Lyra 2/10/19

'''
Answering questions for Gaussian filter:

-   The different sigmas cause different blurring intensities on the images. The larger the sigma value,
    the more blurry the resulting image will be.

-   The median filter removes salt/pepper noise better than the gaussian filter as it will eliminate the
    high and low values of the pixels by picking the median of a block of pixels, but it will distort the 
    edges and cause the image to look "blocky" much more than gaussian, which won't remove the noise as well, 
    but will maintain the edges much better if a proper sized kernel and sigma value is used. The box filter 
    won't remove the noise as well as either the gaussian or median, and it won't keep edges as well as gaussian
    as it will only spread the values of the noise around, actually causing the noise to worsen the pixels around
    it, and also blurring the image while not removing noise.
    
-   you should use median filter with salt/pepper noise images as it will remove the extremes well.
    the box filter will only blur the image, and can sometimes be used on images to approximate the gaussian filter.
    the gaussian filter should be used to smooth images, and can also be used to remove some noise, and it is also
    useful to use when running edge detection algorithms, as it can help identify edges more with a small blur.
'''

from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def GaussianFilter(name, stdDev):
    stdDevSqrd = stdDev**2

    # size for kernel
    size = stdDev * 6
    if size % 2 == 0:
        size += 1

    # opening image in memory
    img = cv2.imread(name)

    # left hand side of 2D gaussian formula
    lhs = 1/(2* math.pi * stdDevSqrd)

    # creating kernel with proper size
    kernel = np.ones((size,size))

    # offset to use for x/y values in kernel
    offset = -(size//2)

    # placing correct values into kernel based on equation
    for i in range(size):
        for j in range(size):
            kernel[i][j] = lhs * (math.exp(-((((offset+i))**2 + ((offset+j))**2 )/(2*stdDevSqrd))) )

    # returning blurred image
    return cv2.filter2D(img, -1, kernel)

# calls gaussian filter function, and plots images
def main2dGaussian():
    
    img1Orig = cv2.imread("image1.png")
    img2Orig = cv2.imread("image2.png")

    plt.subplot(241), plt.imshow(img1Orig), plt.title('Original img1'), plt.xticks([]), plt.yticks([])

    img1 = GaussianFilter("image1.png", 3)
    plt.subplot(242), plt.imshow(img1), plt.title('img1 sigma=3'), plt.xticks([]), plt.yticks([])

    img2 = GaussianFilter("image1.png", 5)
    plt.subplot(243), plt.imshow(img2), plt.title('img1 sigma=5'), plt.xticks([]), plt.yticks([])

    img3 = GaussianFilter("image1.png", 10)
    plt.subplot(244), plt.imshow(img3), plt.title('img1 sigma=10'), plt.xticks([]), plt.yticks([])

    plt.subplot(245), plt.imshow(img2Orig), plt.title('Original img2'), plt.xticks([]), plt.yticks([])

    img4 = GaussianFilter("image2.png", 3)
    plt.subplot(246), plt.imshow(img4), plt.title('img2 sigma=3'), plt.xticks([]), plt.yticks([])

    img5 = GaussianFilter("image2.png", 5)
    plt.subplot(247), plt.imshow(img5), plt.title('img2 sigma=5'), plt.xticks([]), plt.yticks([])

    img6 = GaussianFilter("image2.png", 10)
    plt.subplot(248), plt.imshow(img6), plt.title('img2 sigma=10'), plt.xticks([]), plt.yticks([])

    plt.show()


# call main gaussian function
main2dGaussian()