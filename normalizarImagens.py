import cv2
import numpy as np

img = cv2.imread("img2.jpg")
imgNorm = np.zeros((888, 1152))
imgFinal = cv2.normalize(img, imgNorm, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Imagem 3 normalizada", imgFinal)
cv2.imwrite("img2-normalizada.jpg", imgFinal)
cv2.waitKey(0)
cv2.destroyAllWindows()