import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from math import sqrt, log10
from time import process_time


def compress_with_svd(img_path):
    img = Image.open(img_path)
    img_gray = img.convert('LA')
    img_matrix = np.array(list(img_gray.getdata(band=0)), float)
    img_matrix.shape = (img_gray.size[1], img_gray.size[0])
    img_matrix = np.matrix(img_matrix)
    U, sigma, V = np.linalg.svd(img_matrix)
    return U, sigma, V


def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def test_psnr(img_path, U, sigma, V):
    plt.figure()
    plt.title('Peak Signal-to-Noise Ratio (SVD 2048x2048)')
    plt.xlabel('PSNR')
    plt.ylabel('rank k')
    x, y = [], []
    for i in range(2, 9):
        i = 2**i
        reconstucted_img = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        original = cv2.imread(img_path, 0)
        compressed = np.asarray(reconstucted_img)
        psnr_value = calculate_psnr(original, compressed)
        x.append(psnr_value)
        y.append(i)
    plt.plot(x, y)
    plt.show()


def test_cpu_time(img_path):
    t1_start = process_time()
    compress_with_svd(img_path)
    t1_stop = process_time()
    return t1_stop - t1_start


def test_comrpession_ration(img_path, U, sigma, V):
    original = cv2.imread(img_path, 0)
    original_m = original.shape[0]
    original_size = original_m * original.shape[1]
    for i in range(2, 9):
        i = 2**i
        U_size = U[:, :i].shape[0]
        V_size = V[:i, :].shape[0]
        compressed_size = U_size * i + i + V_size * i
        return round(original_size/compressed_size, 2)


if __name__ == '__main__':
    pass
    # U, sigma, V = compress_with_svd("ex_512.jpg")
    # test_psnr("ex_512.jpg", U, sigma, V)
    # test_comrpession_ration("ex_512.jpg", U, sigma, V)
    # print(test_cpu_time("ex_512.jpg"))

    # U, sigma, V = compress_with_svd("ex_1024.jpg")
    # test_psnr("ex_1024.jpg", U, sigma, V)
    # test_comrpession_ration("ex_1024.jpg", U, sigma, V)
    # print(test_cpu_time("ex_1024.jpg"))

    # U, sigma, V = compress_with_svd("ex_2048.jpg")
    # test_psnr("ex_2048.jpg", U, sigma, V)
    # test_comrpession_ration("ex_2048.jpg", U, sigma, V)
    # print(test_cpu_time("ex_2048.jpg"))
