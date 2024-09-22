# import cv2
# import numpy as np
# x = np.uint8([250])
# y = np.uint8([10])
# print(cv2.add(x,y))
# print(x+y)
# print(np.add(x,y))

import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('./image/cat.jpg')
img2 = cv2.imread('./image/dog.jpg')
img4 = cv2.resize(img2,(500,414))
img3 = cv2.add(img1,img4)
img5 = np.add(img1,img4)
cv2.imshow('cat',img1)
cv2.imshow('dog',img2)
cv2.imshow('add',img3)
cv2.imshow('add',img5)
cv2.waitKey()
cv2.destoryAllWindows()