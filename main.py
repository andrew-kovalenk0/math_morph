from PIL import Image
import numpy as np


def change_border(image_, dx_0, dx_1, dy_0, dy_1, color):
    if color == 'white':
        row = [[255, 255, 255]]
        string = [[255, 255, 255]]
        for _ in range(image_.shape[0] - 1):
            string.append([255, 255, 255])
        for _ in range(image_.shape[1] - 1):
            row.append([255, 255, 255])
        color = np.array([255, 255, 255])
    else:
        row = [[0, 0, 0]]
        string = [[0, 0, 0]]
        for _ in range(image_.shape[0] - 1):
            string.append([0, 0, 0])
        for _ in range(image_.shape[1] - 1):
            row.append([0, 0, 0])
        color = np.array([0, 0, 0])

    row = np.asarray(row)
    string = np.asarray(string)
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


def erase(image_, core_, point, step, border):
    image_buf = image_.copy()
    image_buf = change_border(image_buf, point[0], point[1], core_.shape[0] - point[0] - 1,
                              core_.shape[1] - point[1] - 1, border)
    result_image = image_buf.copy()
    flag = 0
    white_index = []
    for i in range(point[0], image_buf.shape[0] - (core_.shape[0] - point[0] - 1), step):
        for j in range(point[1], image_buf.shape[1] - (core_.shape[1] - point[1] - 1), step):
            for ind_x, x in enumerate(range(i - point[0], i + core_.shape[0] - point[0])):
                for ind_y, y in enumerate(range(j - point[1], j + core_.shape[1] - point[1])):
                    if (core_[ind_x][ind_y] == [255, 255, 255]).all():
                        if(image_buf[x][y] != [255, 255, 255]).all():
                            flag = 1
            if flag == 0:
                white_index.append([i, j])
            else:
                flag = 0
                for ind_x, x in enumerate(range(i - point[0], i + core_.shape[0] - point[0])):
                    for ind_y, y in enumerate(range(j - point[1], j + core_.shape[1] - point[1])):
                        if (core_[ind_x][ind_y] == [255, 255, 255]).all():
                            result_image[x][y] = [0, 0, 0]
    for ind in white_index:
        result_image[ind[0]][ind[1]] = [255, 255, 255]
    result_image = result_image[point[0]:-(core_.shape[0] - point[0] - 1), point[1]:-(core_.shape[1] - point[1] - 1)]
    return result_image


def dilate(image_, core_, point, step):
    result_image = image_.copy()
    for i in range(0, image_.shape[0], step):
        for j in range(0, image_.shape[1], step):
            if (image[i][j] == [255, 255, 255]).all():
                for ind_x, x in enumerate(range(i - point[0], i + core_.shape[0] - point[0])):
                    for ind_y, y in enumerate(range(j - point[1], j + core_.shape[1] - point[1])):
                        if (core_[ind_x][ind_y] == [255, 255, 255]).all():
                            if 0 <= x < image_.shape[0] and 0 <= y < image_.shape[1]:
                                result_image[x][y] = [255, 255, 255]
    return result_image


image_png = Image.open('image.png')
image = np.asarray(image_png)
core_png = Image.open('struct.png')
core = np.asarray(core_png)

dilate_image = dilate(image, core, [1, 1], 1)
dilate_image_png = Image.fromarray(dilate_image)
dilate_image_png.save("dilate_image.png", "PNG")

erase_image = erase(image, core, [1, 1], 1, 'white')
erase_image_png = Image.fromarray(erase_image)
erase_image_png.save("erase_image.png", "PNG")
