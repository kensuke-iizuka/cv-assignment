import numpy as np
import cv2
import matplotlib as plt

def read_image(input_img_name, color_flag=1):
    '''
    返り値はnumpy.ndarray
    '''
    img = cv2.imread(input_img_name, color_flag) 
    return img


def save_image(save_img, file_name):
    cv2.imwrite(file_name, save_img)


def show_image(show_img):
    cv2.imshow("result_image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
