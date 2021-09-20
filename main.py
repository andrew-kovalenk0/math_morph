from PIL import Image
import numpy as np
import time
import cv2
from numba import njit
from scipy import ndimage


@njit
def erode(img, kernel, er_img, bound):
    img_v = img.shape[0]
    img_h = img.shape[1]
    kl_v = kernel.shape[0]
    kl_h = kernel.shape[1]
    kl_cn_v = kl_v // 2
    kl_cn_h = kl_h // 2
    for i in range(kl_cn_v, img_v - (kl_v - kl_cn_v - 1)):
        for j in range(kl_cn_h, img_h - (kl_h - kl_cn_h - 1)):
            er_img[i, j] = np.min(img[i - kl_cn_v: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h][kernel])

    # UP BOUNDARY
    for i in range(kl_cn_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h]
                                          [kernel[kl_cn_v - i:, :]]), bound)
            elif kl_cn_h - 1 >= j:
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, : j + kl_h - kl_cn_h]
                                          [kernel[kl_cn_v - i:, kl_cn_h - j:]]), bound)
            elif j >= img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h:]
                                          [kernel[kl_cn_v - i:, :img_h + kl_cn_h - j]]), bound)

    # DOWN BOUNDARY
    for i in range(img_v - (kl_v - kl_cn_v - 1), img_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h: j + kl_h - kl_cn_h]
                                          [kernel[:kl_cn_v + img_v - i, :]]), bound)
            elif kl_cn_h - 1 >= j:
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, :j + kl_h - kl_cn_h]
                                          [kernel[:kl_cn_v + img_v - i, kl_cn_h - j:]]), bound)

            elif j >= img_h - (kl_h - kl_cn_h - 1):
                er_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h:]
                                          [kernel[:kl_cn_v + img_v - i, :img_h + kl_cn_h - j]]), bound)

    # LEFT BOUNDARY
    for i in range(kl_cn_v):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            er_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, :kl_v - kl_cn_v + i]
                                      [kernel[:, kl_cn_v - i:]]), bound)

    # RIGHT BOUNDARY
    for i in range(img_h - (kl_h - kl_cn_h - 1), img_h):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            er_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, i - kl_cn_h:]
                                      [kernel[:, :kl_h - kl_cn_h + img_h - i - 1]]), bound)

    return er_img


@njit
def dilate(img, kernel, dil_img, bound):
    img_v = img.shape[0]
    img_h = img.shape[1]
    kl_v = kernel.shape[0]
    kl_h = kernel.shape[1]
    kl_cn_v = kl_v // 2
    kl_cn_h = kl_h // 2
    for i in range(kl_cn_v, img_v - (kl_v - kl_cn_v - 1)):
        for j in range(kl_cn_h, img_h - (kl_h - kl_cn_h - 1)):
            dil_img[i, j] = np.max(img[i - kl_cn_v: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h] * kernel)

    # UP BOUNDARY
    for i in range(kl_cn_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h: j + kl_h - kl_cn_h] *
                                           kernel[kl_cn_v - i:, :]), bound)
            elif kl_cn_h - 1 >= j:
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, : j + kl_h - kl_cn_h] *
                                           kernel[kl_cn_v - i:, kl_cn_h - j:]), bound)
            elif j >= img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[: i + kl_v - kl_cn_v, j - kl_cn_h:] *
                                           kernel[kl_cn_v - i:, :img_h + kl_cn_h - j]), bound)

    # DOWN BOUNDARY
    for i in range(img_v - (kl_v - kl_cn_v - 1), img_v):
        for j in range(img_h):
            if kl_cn_h - 1 < j < img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h: j + kl_h - kl_cn_h] *
                                           kernel[:kl_cn_v + img_v - i, :]), bound)
            elif kl_cn_h - 1 >= j:
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, :j + kl_h - kl_cn_h] *
                                           kernel[:kl_cn_v + img_v - i, kl_cn_h - j:]), bound)

            elif j >= img_h - (kl_h - kl_cn_h - 1):
                dil_img[i, j] = max(np.max(img[i - kl_cn_v:, j - kl_cn_h:] *
                                           kernel[:kl_cn_v + img_v - i, :img_h + kl_cn_h - j]), bound)

    # LEFT BOUNDARY
    for i in range(kl_cn_v):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            dil_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, :kl_v - kl_cn_v + i] *
                                       kernel[:, kl_cn_v - i:]), bound)

    # RIGHT BOUNDARY
    for i in range(img_h - (kl_h - kl_cn_h - 1), img_h):
        for j in range(kl_cn_h, img_v - (kl_v - kl_cn_v - 1)):
            dil_img[j, i] = max(np.max(img[j - kl_cn_h: j + kl_h - kl_cn_h, i - kl_cn_h:] *
                                       kernel[:, :kl_h - kl_cn_h + img_h - i - 1]), bound)

    return dil_img


if __name__ == '__main__':
    start_time = time.time()
    image = cv2.imread('image_3.png', 0)
    core = ndimage.generate_binary_structure(2, 1)

    dilate_image = image.copy()
    dilate_image = dilate(image, core, dilate_image, 0)
    Image.fromarray(dilate_image).save("dilate_image.png", "PNG")

    erode_image = dilate_image.copy()
    erode_image = erode(dilate_image, core, erode_image, 0)
    Image.fromarray(erode_image).save("erode_image.png", "PNG")

    print(f'{(time.time() - start_time)} seconds')
