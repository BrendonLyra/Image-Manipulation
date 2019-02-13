# Gradient Operations. Question 4
# Brendon Lyra 2/10/19

from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

# image is (275,183) (width, height)
# derivative masks found in Lec 4 (7/66)

def GradientOperation(name, xKernel, yKernel):

    # opening image in memory, and making it an array
    img = Image.open(name)
    col, row = img.size  # width, height

    # convolve the x and y masks with image with cv2 library
    img = cv2.imread(name)
    xConv = cv2.filter2D(img, -1, xKernel)
    yConv = cv2.filter2D(img, -1, yKernel)

    # 3D array for rgb values. Computing magnitude of x and y images
    magnitude = np.ndarray((row, col, 3))
    for i in range(row):
        for j in range(col):
            for k in range(3):
                magnitude[i][j][k] =  int(math.sqrt((xConv[i][j][k]**2) + (yConv[i][j][k]**2)))
                # make sure values are in valid range
                if magnitude[i][j][k] > 255:
                    magnitude[i][j][k] = 255

    # return x y and magnitude gradient images
    return xConv, yConv, magnitude.astype(int)


# controller function
def mainGradient():

    # imgOrig = cv2.imread("image3.png")
    # # original image
    # plt.subplot(331), plt.imshow(imgOrig), plt.title('Original img3'), plt.xticks([]), plt.yticks([])

    # central gradient
    xCentral, yCentral, magCentral = GradientOperation("image3.png", np.array([-1, 0, 1]), np.array([[-1], [0], [1]]))

    plt.subplot(331), plt.imshow(xCentral), plt.title('x-Central'), plt.xticks([]), plt.yticks([])
    plt.subplot(332), plt.imshow(yCentral), plt.title('y-Central'), plt.xticks([]), plt.yticks([])
    plt.subplot(333), plt.imshow(magCentral), plt.title('magnitude-Central'), plt.xticks([]), plt.yticks([])

    # forward gradient
    xForward, yForward, magForward = GradientOperation("image3.png", np.array([1,-1]), np.array([[1], [-1]]))

    plt.subplot(334), plt.imshow(xForward), plt.title('x-Forward'), plt.xticks([]), plt.yticks([])
    plt.subplot(335), plt.imshow(yForward), plt.title('y-Forward'), plt.xticks([]), plt.yticks([])
    plt.subplot(336), plt.imshow(magForward), plt.title('magnitude-Forward'), plt.xticks([]), plt.yticks([])

    # backward gradient
    xBackward, yBackward, magBackward = GradientOperation("image3.png", np.array([-1,1]), np.array([[-1], [1]]))

    plt.subplot(337), plt.imshow(xBackward), plt.title('x-Backward'), plt.xticks([]), plt.yticks([])
    plt.subplot(338), plt.imshow(yBackward), plt.title('y-Backward'), plt.xticks([]), plt.yticks([])
    plt.subplot(339), plt.imshow(magBackward), plt.title('magnitude-Backward'), plt.xticks([]), plt.yticks([])

    plt.show()

# call to main function of file
mainGradient()