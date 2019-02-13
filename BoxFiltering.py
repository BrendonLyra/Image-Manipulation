# Box Filtering. Question 1
# Brendon Lyra 2/10/19

''' 
Answering question for Box filter:

The larger the kernel size for the box filter, the more blurry the convolution 
between the kernel and image makes the resulting image, but it also slightly
removes some noise from the image, but also blurs the edges. 
'''

from PIL import Image
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
import cv2


def BoxFilter(size, name):
    
    # opening image in memory, and making it an array
    img = cv2.imread(name)
    
    # kernel is 2D array of size "size x size" with all values being 1/size*size
    kernel = np.ones((size, size), np.float32)/np.power(size,2)

    # convolves our "kernel" onto the image. 
    dst = cv2.filter2D(img, -1, kernel)
    
    # return image for plotting
    return dst


def MainBox():
    #Box Filtering with 3x3 and 5x5 kernel sizes
    img1Orig = cv2.imread("image1.png")
    img2Orig = cv2.imread("image2.png")

    plt.subplot(231), plt.imshow(img1Orig), plt.title('Original img1'), plt.xticks([]), plt.yticks([])

    img1 = BoxFilter(3, "image1.png")
    plt.subplot(232), plt.imshow(img1), plt.title('img1 size=3'), plt.xticks([]), plt.yticks([])

    img2 = BoxFilter(5, "image1.png")
    plt.subplot(233), plt.imshow(img2), plt.title('img1 size=5'), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(img2Orig), plt.title('Original img2'), plt.xticks([]), plt.yticks([])

    img3 = BoxFilter(3, "image2.png")
    plt.subplot(235), plt.imshow(img3), plt.title('img2 size=3'), plt.xticks([]), plt.yticks([])

    img4 = BoxFilter(5, "image2.png")
    plt.subplot(236), plt.imshow(img4), plt.title('img2 size=5'), plt.xticks([]), plt.yticks([])

    plt.show()


MainBox()