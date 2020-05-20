import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt, log10
from time import process_time
import scipy
from PIL import Image
from numpy import r_
import scipy.fftpack


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def compress_with_dct(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('LA')
    img_matrix = np.array(list(img_gray.getdata(band=0)), float)
    img_matrix.shape = (img_gray.size[1], img_gray.size[0])
    img_matrix = np.matrix(img_matrix)
    imsize = img_matrix.shape
    img_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            img_dct[i:(i + 8), j:(j + 8)] = dct2(img_matrix[i:(i + 8), j:(j + 8)])
    return img_dct, imsize


def test_cpu_time(img_path):
    t1_start = process_time()
    compress_with_dct(img_path)
    t1_stop = process_time()
    return t1_stop - t1_start


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def test_psnr(img_path, img_dct, imsize):
    plt.figure()
    plt.title('Peak Signal-to-Noise Ratio (DCT 2048x2048)')
    plt.xlabel('PSNR')
    plt.ylabel('coef %')
    x, y = [], []
    for t in range(10,80,10):
        thresh = t/5000
        dct_thresh = img_dct * (abs(img_dct) > (thresh * np.max(img_dct)))
        percent_nonzeros = np.sum(dct_thresh != 0.0) / (imsize[0] * imsize[1] * 1.0)
        y.append(round(percent_nonzeros*100.0, 2))

        img_idct = np.zeros(imsize)
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                img_idct[i:(i + 8), j:(j + 8)] = idct2(dct_thresh[i:(i + 8), j:(j + 8)])
        original = cv2.imread(img_path, 0)
        compressed = np.asarray(img_idct)
        x.append(calculate_psnr(original, compressed))
    plt.plot(x, y)
    plt.show()


def test_comrpession_ration(img_dct, img_size):
    for i in range(10, 80, 10):
        thresh = i / 3220
        dct_thresh = img_dct * (abs(img_dct) > (thresh * np.max(img_dct)))
        cr = (img_size[0] * img_size[1] * 1.0) / np.sum(dct_thresh != 0.0)
        return cr


if __name__ == '__main__':
    pass
    # print("Time for DCT of image of size: ")
    # print("512x512 " + str(test_cpu_time("ex_512.jpg")))
    # print("1024x1024 " + str(test_cpu_time("ex_1024.jpg")))
    # print("2048x2048 " + str(test_cpu_time("ex_2048.jpg")))


    # dct_img, im_size = compress_with_dct("ex_512.jpg")
    # test_psnr("ex_512.jpg", dct_img, im_size)
    # dct_img, im_size = compress_with_dct("ex_1024.jpg")
    # test_psnr("ex_1024.jpg", dct_img, im_size)
    # dct_img, im_size = compress_with_dct("ex_2048.jpg")
    # test_psnr("ex_2048.jpg", dct_img, im_size)



