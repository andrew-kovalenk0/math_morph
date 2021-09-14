from PIL import Image
import numpy as np
import time
import cv2
from numba import njit
from scipy import ndimage


def change_border(image_, dx_0, dx_1, dy_0, dy_1, color):
    row = np.asarray([color] * image_.shape[0])
    string = np.asarray([color] * image_.shape[1])

    for _ in range(dx_0):
        image_ = np.insert(image_, 0, string, 0)
        row = np.insert(row, 0, color, 0)

    for _ in range(dy_0):
        image_ = np.insert(image_, 0, row, 1)
        string = np.insert(string, 0, color, 0)

    for _ in range(dx_1):
        image_ = np.insert(image_, image_.shape[0], string, 0)
        row = np.insert(row, 0, color, 0)

    for _ in range(dy_1):
        image_ = np.insert(image_, image_.shape[1], row, 1)
    return image_


@njit
def dilate(img, kernel, boun, dil_img):
    img_v = img.shape[0]
    img_h = img. shape[1]
    kl_v = kernel.shape[0]
    kl_h = kernel.shape[1]
    kl_cn_v = kl_v // 2
    kl_cn_h = kl_h // 2
    for i in range(kl_cn_v, img_v - (kl_v - kl_cn_v - 1)):
        for j in range(kl_cn_h, img_h - (kl_h - kl_cn_h - 1)):
            dil_img[i - kl_cn_v, j - kl_cn_h] = np.max(img[i - kl_cn_v: i + kl_v - kl_cn_v,
                                                           j - kl_cn_h: j + kl_h - kl_cn_h] * kernel)
    return dil_img


# -------------MAIN-------------- #
image = cv2.imread('image.png', 0)
core = ndimage.generate_binary_structure(2, 1).astype('int')
image = change_border(image, core.shape[0] // 2, core.shape[1] // 2,  core.shape[0] - core.shape[0] // 2 - 1,
                      core.shape[0] - core.shape[1] // 2 - 1, 0)

dilate_image = image.copy()
start_time = time.time()
dilate_image = dilate(image, core, 0, dilate_image)
print(f'{(time.time() - start_time)} seconds')
dilate_image_png = Image.fromarray(dilate_image)
dilate_image_png.save("dilate_image.png", "PNG")

# erode_image = erode(image, core, [2, 2], 1, 'white')
# erode_image_png = Image.fromarray(erode_image)
# erode_image_png.save("erode_image.png", "PNG")
