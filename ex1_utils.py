"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
normalization_factor = 256
GS = 1
RGB = 2
EPSILON = 0.0001

YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])

# inverse of YIQ matrix
RGB_MAT = np.array([[1.00000001, 0.95598634, 0.6208248],
                    [0.99999999, -0.27201283, -0.64720424],
                    [1.00000002, -1.10674021, 1.70423049]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315820019


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv.imread(filename, -1)
    if representation is RGB:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if representation is GS and len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = img.astype(np.float) / normalization_factor
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.gray()
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # compute with YIQ matrix
    ans = imgRGB.dot(YIQ_MAT.T)

    return ans

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # compute with inverse matrix
    ans = imgYIQ.dot(RGB_MAT.T)

    return ans


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
            Equalizes the histogram of an image
            :param imgOrig: Original Histogram
            :ret
        """
    cop = np.copy(imgOrig)
    channel = cop
    # if RGB - transform to YIQ and equalize the Y channel
    if len(imgOrig.shape) > 2:
        cop = transformRGB2YIQ(cop)
        channel = cop[:, :, 0]
    # denormalize
    channel = (channel * 255).astype(np.int)
    channel, histo, new_histo = channel_equalize(channel)
    channel /= 255
    # if RGB reform imgs back with the new Y channel
    if len(imgOrig.shape) > 2:
        cop[:, :, 0] = channel
        cop = transformYIQ2RGB(cop)
    if len(imgOrig.shape) <= 2:
        cop = channel
    return cop, histo, new_histo


def channel_equalize(channel):
    histo, _ = np.histogram(channel, bins=256, range=(0, 255))
    cumsum = np.cumsum(histo)
    max_val = np.max(channel)
    num_total_pixels = cumsum[255]
    # create look up table for mapping the values to the new values
    lu_table = [np.ceil(max_val * m / num_total_pixels) for m in cumsum]
    ans = np.zeros_like(channel, dtype=np.float)
    # map
    for i in range(len(ans)):
        for j in range(len(ans[0])):
            ans[i][j] = lu_table[channel[i][j]]

    new_histo, _ = np.histogram(ans, bins=256, range=(0, 255))

    return ans, histo, new_histo


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
            Quantized an image in to **nQuant** colors
            :param imOrig: The original image (RGB or Gray scale)
            :param nQuant: Number of colors to quantize the image to
            :param nIter: Number of optimization loops
            :return: (List[qImage_i],List[error_i])
        """
    # not RGB - no need to YIQ transform
    if len(imOrig.shape) <= 2:
        img = np.copy(imOrig)
        return quantize(img, nQuant, nIter)
    # RGB img - more efficient to transform to YIQ and quantize Y channel
    yiq = transformRGB2YIQ(imOrig)
    y = yiq[:, :, 0]
    i = yiq[:, :, 1]
    q = yiq[:, :, 2]
    img = np.copy(y)
    qImage, errs = quantize(img, nQuant, nIter)
    imgs = []
    # reform images with Y qunatized channel
    for img in qImage:
        imgs.append(transformYIQ2RGB(np.dstack((img, i, q))))
    return imgs, errs


def quantize(channel, nQuant, nIter):
    # denormalize
    channel *= normalization_factor
    channel = channel.astype(np.int)
    hist, _ = np.histogram(channel, bins=256, range=(0, 255))
    borders = np.zeros(nQuant + 1).astype(np.int)
    # borders by the amounts
    compute_first_borders(hist, nQuant + 1, borders)
    errs = []
    imgs = []
    # start quantize n times following alternating optimization algorithm
    # alternating between computing means by borders and computing border by means
    for iter in range(nIter):
        means = compute_means(borders, nQuant, hist)
        for i in range(len(means)):
            if math.isnan(means[i]):
                means[i] = 0
        ith_image = np.zeros_like(channel)
        for k in range(nQuant):
            ith_image[channel > borders[k]] = means[k]

        # compute err and add results
        errs.append(np.sqrt((channel - ith_image) ** 2).mean())
        imgs.append(ith_image / normalization_factor)
        compute_new_borders(borders, means)
    return imgs, errs


def compute_means(borders, n_quant, hist):
    means = []
    # compute mean for each window, using weighted average
    # need mean intensity
    # weights - number of pixels in the intensity
    for k in range(n_quant):
        window = hist[borders[k]: borders[k + 1]]
        weighted = window * range(len(window))
        m = (weighted.sum() / (np.sum(window) +EPSILON)+ borders[k] )
        means.append(m)
    means = np.array(means).astype(np.int)
    fix_nan(means)
    return means


def compute_new_borders(borders, means):
    # ith border = mean of mean[i-1] and mean[i]
    for i in range(len(means) - 1):
        borders[i + 1] = np.mean([means[i], means[i + 1]])

def fix_nan(arr):
    for i in range(len(arr)):
        if math.isnan(arr[i]):
            arr[i] = 0

def compute_first_borders(arr, num_windows, borders):
    s = np.sum(arr)
    # computing the first borders by amounts of pixels in intensities
    window_size = np.ceil((np.sum(arr) / num_windows))
    borders[0] = 0
    k = 1
    summer = 0
    for i in range(len(arr)):
        summer += arr[i]
        if summer >= window_size:
            borders[k] = i
            k += 1
            summer = 0

