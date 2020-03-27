#coding=utf-8

import cv2
import numpy as np

#sigma - Padrão do envelope gaussiano (Altera a quantidade de elipses procuradas) = 10
#teta - Orientação (inclinação em graus) = 0
#lambda - Comprimento de onda (pensando em 2d, altera a variação x0 e x) = 30
#psi - Deslocamento da função senoide (Desloca a/as elipse/elipses) = 0
#gama - Proporção espacial e elipticidade da função(pensando em 2d, altera a elipse que procuramos, em tamanho y0 e y) = 0.25
def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 1  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def criarBancoDeFiltros():
    filtros = []

    for theta in np.arange(0, np.pi, np.pi/18):
        parametros = {'sigma':1.0, 'theta':theta, 'Lambda':10,
                  'psi':2, 'gamma':0.5}
        kern = gabor(**parametros)
        kern /= 1.5 * kern.sum()
        filtros.append((kern, parametros))

    return filtros

def processarImagem(img, filtros):
    imgFinal = np.zeros_like(img)

    for kern, parametros in filtros:
        imgFiltrada = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(imgFinal, imgFiltrada, imgFinal)

    return imgFinal


#main
filtros = criarBancoDeFiltros()

img = cv2.imread('img2-normalizada.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagem', img)

imagemFiltrada = processarImagem(img, filtros)

cv2.imshow('imagem filtrada', imagemFiltrada)
cv2.waitKey(0)
cv2.destroyAllWindows()