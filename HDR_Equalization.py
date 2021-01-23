import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

imArr = np.array([np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_01.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_02.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_03.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_04.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_05.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_06.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_07.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_08.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_09.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_10.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_11.jpg")),
np.array(Image.open("HDR02/HDRI_Sample_Scene_Window_-_12.jpg")),])

imgsOut = np.average(imArr, axis=0)
outImg = Image.fromarray(np.uint8(imgsOut))
outImg.save("output.png")


path = "output.png"
img = cv2.imread(path)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Histogram
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

R, G, B = cv2.split(img)

output1_R = cv2.equalizeHist(R)
output1_G = cv2.equalizeHist(G)
output1_B = cv2.equalizeHist(B)

equ = cv2.merge((output1_R, output1_G, output1_B))

cv2.imshow('equ.png',equ)
cv2.imwrite('equ.png',equ)

cv2.waitKey(0)
cv2.destroyAllWindows()

hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()