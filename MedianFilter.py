# Median Filtering. Question 2
# Brendon Lyra 2/10/19


from PIL import Image
from scipy import ndimage
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt



# will not be processing problem edges, in which box can be out of bounds.
# will leave those pixels as they are
def MedianFilter(name, size):
    
    # open image in memory, and obtaining size
    img = Image.open(name)
    col, row = img.size

    # convert image to 2d array, instead of rgb values, since image is gray
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.array(img, dtype=int)

    print(img)

    finalImg = np.ndarray((row, col))
    
    # to sort values and chose middle one
    medianArray = np.zeros(size**2)

    # to size of kernel to pick values from
    # kernel = np.ndarray((size,size))

    # move to the left by size/2
    # move up by size/2

    # move to the right by size
    # then move down one

    # loop through whole image (i/j)
    for i in range(row):
        for j in range(col):
            #print(img[i][j])

            # for each pixel, pick median value and put that in new image (unless there is a boundary error)
            # loop through kernel (k/z)
            try:
                for k in range(size):
                    for z in range(size):
                        count = 0
                        # place all values in kernel into median array
                        temp = img[i-int(size//2)+k][j-int(size//2)+z]
                        #print(temp)
                        medianArray[count] = temp
                        count +=1
            except:
                None

            medianArray.sort()
            #print(medianArray)
            finalImg[i][j] = medianArray[int(size/2)]
    
    #print (finalImg.astype(int))
    return finalImg
                




# main parent function
def MainMedian():
    
    im = MedianFilter("image2.png", 7)

    plt.subplot(111), plt.imshow(im, cmap="gray"), plt.title('Median filter'), plt.xticks([]), plt.yticks([])
    plt.show()


MainMedian()