import scipy.ndimage as I
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt


def keypress():
  ESC = 27
  while True:
    keycode = cv2.waitKey(25)
    if keycode != -1:
      keycode &= 0xFF
      if keycode == ESC:
        break
  cv2.destroyAllWindows()

#### Imagem tons de cinza ######
img_bgr = cv2.imread('quadrados.png')
cv2.imshow('Original', img_bgr)

keypress()

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

cv2.imshow('Tons de Cinza', img_gray)
 
keypress()
################################

###### FILTRO PASSA-ALTA #####

kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  1,  2,  1, -1],
    [-1,  2,  4,  2, -1],
    [-1,  1,  2,  1, -1],
    [-1, -1, -1, -1, -1]
])

k5_1 = cv2.filter2D(img_gray, -1, kernel_5x5)
 
cv2.imshow("Filtro passa-alta", k5_1)

keypress()
########################################

############ BINARIZAÇÃO DA IMAGEM (PRETO E BRANCO) #################
(thresh, im_bw) = cv2.threshold(k5_1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(k5_1, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Imagem Binarizada", im_bw)

keypress()
#################################################

############## Transformada de Hough ################
edges = cv2.Canny(img_gray,50,150,apertureSize = 3)
print(img_gray.shape[1])
print(img_gray.shape)
minLineLength=img_bgr.shape[1]-300
lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)

a,b,c = lines.shape
for i in range(a):
    cv2.line(img_bgr, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (50, 205, 153), 3, cv2.LINE_AA)


cv2.imshow('edges', edges)
cv2.imshow('result', img_bgr)

keypress()

#####################################################

