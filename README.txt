Image Representations and Point Operations:
Python Version - 3.8.5
Developing and Testing Platform - Jetbrains' Pycharm IDE
Files:
1.Ex1.pdf - Assignment Instructions
2.ex1_main.py - Test Script
3.ex1_utils.py - Functions of Image Processing Algorithms
4.gamma.py - gamma correctiong GUI
5.bac_con.png - image for testing supplied by T.A.
6.beach.jpg - image for testing supplied by T.A.
7.dark.jpg - image for testing supplied by T.A.
8.water_bear.png - image for testing supplied by T.A.
9.testImg1.jpg - image of Yosemite Park for self testing
9.testimg2.jpg - image of Slash for self testing

Functions in the project:
ex1_utils:
imReadAndConvert - Function for loading an img from a certain path in a specific mode (RGB/CS)
imDisplay - loading and displaying img from a certain path in a specific mode
transformRGB2YIQ - convert an img from RGB representation to YIQ
transformYIQ2RGB - convert an img from YIQ representation to RGB
hsitogramEqualize - histogram equalize an img
channel_equalize - help method for hsitogramEqualize for equalizing a channel
qunatizeImage - img qunatization for img 
qunatize - help method for quantizeImage for quantization of a channel
compute_means - help method for quantize, computing means of each window
compue_new_borders - help method for quantize, computing borders of windows according to means
compute_first_borders  - help method for quantize, computing first borders by amounts of pixels in intensities

gamm.py:
gammaDisplay - function for initialize window and trackbar

