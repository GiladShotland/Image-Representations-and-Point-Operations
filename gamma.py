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
from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv
import numpy as np
gamma_max = 200
gamma_range = 10
title_window = 'Gamma Correction'

def trackbar(val):
    gma = gamma_range * (val / gamma_max)
    # gamma requested
    print("\rGamma {}".format(gma), end='')
    # gamma correction
    dst = ((img / 255) ** (gma) * 255).astype(np.uint8)
    cv.imshow(title_window, dst)

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # for indentifying in trackbar func
    global img
    img = cv.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.namedWindow(title_window)
    trackbar_name = 'Gamma'
    cv.createTrackbar(trackbar_name, title_window, 0, gamma_max, trackbar)
    trackbar(0)
    cv.waitKey()
    cv.destroyAllWindows()



def main():
    gammaDisplay('water_bear.png', 2)


if __name__ == '__main__':
    main()
