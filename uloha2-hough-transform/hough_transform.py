#!/usr/bin/env python3
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math, sys


def grayscale(rgb):
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    y = [0.299, 0.587, 0.114]
    res = y[0] * rgb[:, :, 0] + y[1] * rgb[:, :, 1] + y[2] * rgb[:, :, 2]
    return res

def convolution(img, kernel, ker_size):
    res = np.zeros_like(img)
    added = int(ker_size / 2)
    # Pre prejdenie vsetkych pixelov, aj hran, zvacsime povodne rozmery obrazka
    image_tmp = np.zeros((img.shape[0] + added * 2, img.shape[1] + added * 2))
    image_tmp[added:-added, added:-added] = img
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            res[i, j] = (image_tmp[i: i + ker_size, j: j + ker_size] * kernel).sum()
    return res

#https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
def non_max_suppression(img, D):
    Z = np.zeros_like(img)
    angle = D * 180.0 / 3.14159265
    angle[angle < 0] += 180
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = img[i + 1, j]
                r = img[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            else:
                q = 255
                r = 255

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0
    return Z

def double_threshold(img, low=0.05, high=0.25, weak=5, strong=255):
    high = img.max() * high
    low = high * low
    res = np.zeros_like(img)
    s_i, s_j = np.where(img >= high)
    w_i, w_j = np.where((img <= high) & (img >= low))
    res[w_i, w_j] = weak
    res[s_i, s_j] = strong
    return res

def edge_tracking_by_hysteresis(img, weak, strong):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == weak:
                if (img[i, j - 1] == strong
                        or img[i, j + 1] == strong
                        or img[i + 1, j - 1] == strong
                        or img[i + 1, j] == strong
                        or img[i + 1, j + 1] == strong
                        or img[i - 1, j - 1] == strong
                        or img[i - 1, j] == strong
                        or img[i - 1, j + 1] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detector(img, low_threshold, high_threshold):
    # 1. krok
    # Gaussian kernel
    x = np.array([[-2, -2, -2, -2, -2],
                  [-1, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2]])
    y = np.array([[-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2],
                  [-2, -1, 0, 1, 2]])
    # https://en.wikipedia.org/wiki/Gaussian_blur
    ou = 0.84089642
    gk = (1 / (2 * 3.14159265 * ou ** 2)) * np.exp(-((x ** 2 + y ** 2) / (2 * (ou ** 2))))
    img_1 = convolution(img, gk, 5)

    # 2. krok
    # Gradient intenzita
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Ix = convolution(img_1, Kx, 3)
    Iy = convolution(img_1, Ky, 3)
    g = np.hypot(Ix, Iy)
    img_2 = g / g.max() * 255
    theta = np.arctan2(Iy, Ix)

    # 3. krok
    # Non-maximum potlacenie - stensenie objavenych hran
    img_3 = non_max_suppression(img_2, theta)

    # 4. krok
    # Double threshold
    weak_points = 5
    strong_points = 255
    img_4 = double_threshold(img_3, low_threshold, high_threshold, weak_points, strong_points)

    # 5. krok
    # Hystereza - detekcia hran
    img_5 = edge_tracking_by_hysteresis(img_4, weak_points, strong_points)
    return img_5

def hough_transformation(img, width, height, value_threshold=5):
    thety = np.deg2rad(np.arange(-90.0, 90.0, 1))
    num_thety = len(thety)
    cos_thety = np.cos(thety)
    sin_thety = np.sin(thety)
    uhl_dlzka = int(round(math.sqrt(width**2 + height**2)))
    rka = np.linspace(-uhl_dlzka, uhl_dlzka, uhl_dlzka * 2)
    # Inicializacia akumulatora
    accumulator = np.zeros((2 * uhl_dlzka, num_thety))
    edges = img > value_threshold
    y_idxs, x_idxs = np.nonzero(edges)
    # Volenie pre jednotlive hodnoty do akumulatora
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thety):
            r = uhl_dlzka + int(round(x * cos_thety[t_idx] + y * sin_thety[t_idx]))
            accumulator[r, t_idx] += 1
    return accumulator, thety, rka

def get_hough_lines(accumulator, thety, rka, image_width, threshold=100):
    i, j = np.where(accumulator > threshold)
    r = rka[i]
    theta = thety[j]
    xvals = []
    yvals = []
    for ind in range(0, theta.size):
        xvals.append(np.arange(0, image_width))
        yvals.append((r[ind] - xvals[ind] * np.cos(theta[ind])) / np.sin(theta[ind]))
    return xvals, yvals, i, j

def show_image(img, name, save_to_path=None):
    plt.imshow(img)
    plt.title(name)
    if img.ndim == 2:
        plt.set_cmap('gray')
    if save_to_path is not None:
        plt.savefig(save_to_path)
    plt.show()

def show_image_with_detected_lines(img, name, xvals, yvals, save_to_path=None):
    plt.imshow(img)
    plt.title(name)
    plt.axis('image')
    for i in range(0, np.shape(xvals)[0]):
        plt.plot(xvals[i], yvals[i], color='green')
    if save_to_path is not None:
        plt.savefig(save_to_path)
    plt.show()

def show_detected_lines(img, name, accumulator, acc_i, acc_j, xvals, yvals, save_to_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title(name)
    ax[0].axis('image')
    ax[0].set_xlim((0, img.shape[1]))
    ax[0].set_ylim((img.shape[0], 0))
    for i in range(0, np.shape(xvals)[0]):
        ax[0].plot(xvals[i], yvals[i], color='red')
    ax[1].imshow(accumulator, cmap=plt.cm.terrain)
    for i in range(0, np.size(acc_i)):
        ax[1].plot(acc_j[i], acc_i[i], color='red', marker='*', markersize=4)
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()

def test_image(img, acc_threshold, output_name, thres_low, thres_high, save_to_path=None):
    w = img.shape[0]
    h = img.shape[1]
    img_grey = grayscale(img)
    img_with_edges = canny_edge_detector(img_grey, thres_low, thres_high)
    accumulator, thety, rka = hough_transformation(img_with_edges, w, h)
    xvals, yvals, acc_i, acc_j = get_hough_lines(accumulator, thety, rka, w, acc_threshold)
    show_detected_lines(img_with_edges, output_name, accumulator, acc_i, acc_j, xvals, yvals, save_to_path)
    return xvals, yvals, img_with_edges

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: hough_transform.py <path_to_picture> <hough_transform_threshold>')
    else:
        img_path = sys.argv[1]
        hough_threshold = int(sys.argv[2])
        img = mpimg.imread(img_path)
        xlines, ylines, img_with_edges = test_image(img, hough_threshold, 'Hough transform detekcia', 0.02, 0.10, 'Hough_transform_detekcia')
        show_image(img_with_edges, "Obraz - detekcia hran", "Obr_edges")
        show_image_with_detected_lines(img, "Vstupny obraz - vysledok", xlines, ylines, 'obr_final')


    ## Nase testovacie scenare
    #pentagon_img = mpimg.imread('test_images/pentagon.jpg')
    #window_img = mpimg.imread('test_images/Window.jpg')
    #sudoku_img = mpimg.imread('test_images/sudoku.jpg')

    #xlines, ylines, img_with_edges = test_image(pentagon_img, 100, 'Pentagon Image', 0.05, 0.75, 'Hough_pentagon')
    #show_image(img_with_edges, "Pentagon - detekcia hran", "Pentagon_edges")
    #show_image_with_detected_lines(pentagon_img, "Pentagon - vysledok", xlines, ylines, 'Pentagon_final')

    #xlines, ylines, img_with_edges = test_image(sudoku_img, 150, 'Sudoku Image', 0.02, 0.10, 'Hough_Sudoku')
    #show_image(img_with_edges, "Sudoku - detekcia hran", "Sudoku_edges")
    #show_image_with_detected_lines(sudoku_img, "Sudoku - vysledok", xlines, ylines, 'Sudoku_final')

    #xlines, ylines, img_with_edges = test_image(window_img, 80, 'Window Image', 0.05, 0.45, 'Hough_Window')
    # show_image(img_with_edges, "Window - detekcia hran", "Window_edges")
    #show_image_with_detected_lines(window_img, "Window - vysledok", xlines, ylines, 'Window_final')