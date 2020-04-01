# opencv
import cv2
import numpy as np

def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':4.5,
                  'psi':0, 'gamma':0.25 , 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1 *kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

#main
filters = build_filters()

img = cv2.imread('img3-normalizada.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', img)

filtered_img = process(img, filters)

cv2.imshow('filtered image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()