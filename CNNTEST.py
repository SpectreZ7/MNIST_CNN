from . import CNNTRAIN
import cv2
img = cv2.imread('path_to_digit_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))


img = img.astype('float32') / 255.0

img = img.reshape(1, 28, 28, 1)
import matplotlib as plt