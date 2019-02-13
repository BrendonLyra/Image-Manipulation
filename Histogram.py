# Histogram function
# Brendon Lyra 2/10/19


from PIL import Image
from scipy import ndimage
import numpy as np
import math
import matplotlib.pyplot as plt



def Histogram(name, numBins):

    # opening image in memory, and making it an array
    img = Image.open(name)
    col, row = img.size
    img_arr = np.array(img, dtype=float)

    # will hold number of pixels for each value
    freq = np.zeros(256)
    freq = freq.astype(int)

    # will hold what array value it is in. For plotting purposes, on x-direction.
    xArr = np.zeros(numBins)
    xArr = xArr.astype(int)

    for i in range (numBins):
        xArr[i] = i

    # add +1 to each value we see a pixel for
    for i in range(row):
        for j in range(col):
            freq[int(img_arr[i][j][0])] += 1

    # creating frequency in regards to num of bins
    binsFreq = np.zeros(numBins)
    numOverBins = 256/numBins

    # go through the old array and plug in correct values into new array corresponding to correct bin number
    binNum = 0
    for i in range (256):
        if (i % numOverBins == 0) and (i != 0):
            binNum += 1
        binsFreq[binNum] += freq[i]

    # plot and show bar graph/histogram
    plt.bar(xArr,binsFreq)
    plt.show()


# main controller function 
def MainHistogram():

    # image is (788,662) (width, height).    
    name = "image4.png"
    
    # numBins = 256, 128, 64
    
    Histogram(name, 256)
    Histogram(name, 128)
    Histogram(name, 64)


MainHistogram()