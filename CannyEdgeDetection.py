# Canny Edge algorithm implementation. Last question
# Brendon Lyra 2/10/19


'''
Answering questions for Canny edge detection:

-   The lower the sigma value, the better and more defined the edges seemed to be in the final image.
    With sigma =1 the edges are well defined, and not thick, but with sigma=10 the edges are very thick,
    and not well defined, making sigma=1 the better option since it is the lowest value.

-   The fast gaussian filter works best for the Canny edge detection algorithm shown below, especially with
    the lower sigma value of 1. The lower sigma proved to be better for edge detection in both canny1/canny2.jpg
    images used as input. the low/high threshold values proved to also be the best for edge detection in both 
    input images used for this algorithm.

'''

from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy, copy

# img_arr[i][j][0] = r
# img_arr[i][j][1] = g
# img_arr[i][j][2] = b
# img_arr[i][j][3] = 255


def CannyAlgorithm(name, sigma):

    gaussianString = 'Gaussian sigma=' + str(sigma)
    finalString = "Final image, sigma=" + str(sigma)

    # opening image in memory, and making it an array
    img = Image.open(name)
    col, row = img.size  # width, height
    
    # re-open image with cv2 library
    imgcv2 = cv2.imread(name)

    # plotting original image
    plt.subplot(331), plt.imshow(imgcv2), plt.title('Original'), plt.xticks([]), plt.yticks([])

    fGaussianImg = FastGaussian(name, sigma)
    plt.subplot(332), plt.imshow(fGaussianImg, cmap="gray"), plt.title(gaussianString), plt.xticks([]), plt.yticks([])
    
    xGrad, yGrad, gradDir, magGrad, gradientImg = Gradient(fGaussianImg, name)
    plt.subplot(333), plt.imshow(xGrad, cmap="gray"), plt.title('x-Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(334), plt.imshow(yGrad, cmap="gray"), plt.title('y-Gradient'), plt.xticks([]), plt.yticks([])
    plt.subplot(335), plt.imshow(gradDir, cmap="gray"), plt.title('gradient-Direction'), plt.xticks([]), plt.yticks([])
    plt.subplot(336), plt.imshow(magGrad, cmap="gray"), plt.title('gradient-Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(337), plt.imshow(gradientImg, cmap="gray"), plt.title('non-max Suppression'), plt.xticks([]), plt.yticks([])

    plt.show()

    # obtain and show final image, alone
    FinalImg = Thresholding(row, col, gradientImg)
    plt.subplot(121), plt.imshow(FinalImg, cmap="gray"), plt.title(finalString), plt.xticks([]), plt.yticks([])
    plt.show()
    

def FastGaussian(name, stdDev):

    stdDevSqrd = stdDev**2

    # size of kernel
    size = stdDev * 6
    if size % 2 == 0:
        size += 1

    # opening image in memory, and making it an array
    # img = Image.open(name).convert('LA')
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.array(img, dtype=float)
    
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
    xConvolved = cv2.filter2D(img,-1, xKernel)
    joinedImg = cv2.filter2D(xConvolved,-1, yKernel)

    return joinedImg


def Gradient(FilteredImg, name):
    # opening image in memory, and making it an array
    img = Image.open(name)
    col, row = img.size  # width, height

    # derivative masks found in Lec 4 (9/66). WILL BE TRYING THIS ONE
    xKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # / 9
    yKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # / 9

    # convolve the x and y masks with image with cv2 library
    xConv = cv2.filter2D(FilteredImg, -1, xKernel)
    yConv = cv2.filter2D(FilteredImg, -1, yKernel)

    # 2D array for grey values for magnitude
    magnitude = np.ndarray((row, col))
    for i in range(row):
        for j in range(col):
            # for k in range(3):
            magnitude[i][j] =  int(math.sqrt((xConv[i][j]**2) + (yConv[i][j]**2)))
    
    # copying so as not to have them pointing to same object.
    yCpy = deepcopy(yConv)
    xCpy = deepcopy(xConv)

    # convert to degrees from radians, then from float to int
    gradientDirections = (np.arctan(yCpy, xCpy))*(180/np.pi)

    # convert from negative values to positive degrees, and also round the angles to 0,45,90,135
    for i in range(row):
        for j in range(col):
            
            if gradientDirections[i][j] < 0:
                gradientDirections[i][j] += 180

            if (0 <= gradientDirections[i][j] < 22.5) or (157.5 <= gradientDirections[i][j] < 180):
                    gradientDirections[i][j] = 0

            elif (22.5 <= gradientDirections[i][j] < 67.5):
                    gradientDirections[i][j] = 45
                
            elif (67.5 <= gradientDirections[i][j] < 112.5):
                    gradientDirections[i][j] = 90
                
            else:
                    gradientDirections[i][j] = 135

    thinnedMagnitude = NonMaxSuppression(row, col, deepcopy(magnitude), deepcopy(gradientDirections))

    # returns magnitude with thinned edges
    return xConv, yConv, gradientDirections, magnitude, thinnedMagnitude


# edge thinning with non-max suppression algorithm. Try/catches to stop out-of-bounds checks
def NonMaxSuppression(row, col, magnitude, gradientDirections):
    for i in range(row):
        for j in range(col):

            # 0 angle is right/left check. [0][+1] and [0][-1]
            if gradientDirections[i][j] == 0:
                try: 
                    if (magnitude[i][j] <= magnitude[i][j+1]):
                        magnitude[i][j] = 0
                except:
                    None

                try: 
                    if (magnitude[i][j] <= magnitude[i][j-1]):
                        magnitude[i][j] = 0
                except:
                    None
            
            # 90 angle is up and down check. [+1][0] and [-1][0]    
            if gradientDirections[i][j] == 90:
                try: 
                    if (magnitude[i][j] <= magnitude[i-1][j]):
                            magnitude[i][j] = 0
                except:
                        None

                try: 
                    if (magnitude[i][j] <= magnitude[i+1][j]):
                            magnitude[i][j] = 0
                except:
                        None
            
            # 45 angle is up right/ down left check. [-1][+1] and [+1][-1]
            if gradientDirections[i][j] == 45:
                try: 
                    if (magnitude[i][j] <= magnitude[i-1][j+1]):
                            magnitude[i][j] = 0
                except:
                    None

                try: 
                    if (magnitude[i][j] <= magnitude[i+1][j-1]):
                            magnitude[i][j] = 0
                except:
                    None

            # 135 angle is up left, and down right. and [-1][+1] and [-1][+1]
            if gradientDirections[i][j] == 135:
                try: 
                    if (magnitude[i][j] <= magnitude[i-1][j-1]):
                            magnitude[i][j] = 0
                except:
                    None

                try: 
                    if (magnitude[i][j] <= magnitude[i+1][j+1]):
                            magnitude[i][j] = 0
                except:
                    None

    return magnitude

    
def Thresholding(row, col, img):

    LowThresh = 25
    HighThresh = LowThresh*(1.5)

    lowPix = 0
    HighPix = 255
    UnknownPix = 25 # don't know if it is high or low yet

    locationArr = np.ndarray((row,col))

    # check for values that are higher, lower, or between threshold. Set pixels to corresponding values
    for i in range(row):
        for j in range(col):

            if img[i][j] > HighThresh:
                locationArr[i][j] = HighPix
            
            elif img[i][j]  < LowThresh:
                locationArr[i][j] = lowPix
            
            else:
                locationArr[i][j] = UnknownPix
    
    # to check all 8 borders
    dx = np.array([-1,0,1,-1,1,-1,0,1])
    dy = np.array([-1,-1,-1,0,0,1,1,1])
    check = False

    # check unknown pixels if they are connected to edges or not. Using 8-connected (slide-6 59/72)
    for i in range(row):
        for j in range(col):
            
            if locationArr[i][j] == UnknownPix:
                
                check = False

                # check 8 borders if there is an edge(high) pixel
                for k in range(8):
                    try:
                        if locationArr[dy[k]][dx[k]] == HighPix:
                            # img[i][j] = HighPix
                            check = True
                    except:
                        None
                if check == False:
                    img[i][j] = lowPix

    return img


# main controller function of this file.
def CannyMain():

    sigmas = np.array([1,2,5])

    # loop for different sigmas with first image
    for i in range(3):
        CannyAlgorithm("canny1.jpg", sigmas[i])

    # loop for second image
    for i in range(3):
        CannyAlgorithm("canny2.jpg", sigmas[i])


# will call the local main function in this file.
CannyMain()