from PIL import Image
import numpy as np
import time

dilate_row = 0
dilate_string = 0
dilate_index = []


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
        image_ = np.insert(image_, 0, string, 1)
        row = np.insert(row, 0, color, 0)

    for _ in range(dy_0):
        image_ = np.insert(image_, 0, row, 0)
        string = np.insert(string, 0, color, 0)

    for _ in range(dx_1):
        image_ = np.insert(image_, image_.shape[1], string, 1)
        row = np.insert(row, 0, color, 0)

    for _ in range(dy_1):
        image_ = np.insert(image_, image_.shape[0], row, 0)
    return image_


def erode(image_, core_, point, step, border):
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


def dilate(image_, shape, point_, core_):
    global dilate_string
    global dilate_row
    global dilate_image
    if (image_ == [255, 255, 255]).all():
        for ind_x, x in enumerate(range(dilate_string - point_[0], dilate_string + core_.shape[0] - point_[0])):
            for ind_y, y in enumerate(range(dilate_row - point_[1], dilate_row + core_.shape[1] - point_[1])):
                if (core_[ind_x][ind_y] == [255, 255, 255]).all():
                    if 0 <= x < shape[0] and 0 <= y < shape[1]:
                        dilate_image[x][y] = [255, 255, 255]
    dilate_row += 1
    if dilate_row == shape[1]:
        dilate_row = 0
        dilate_string += 1


# -------------MAIN-------------- #
start_time = time.time()
image_png = Image.open('image.png')
image = np.asarray(image_png)
core_png = Image.open('core.png')
core = np.asarray(core_png)

dilate_image = image.copy()
np.apply_along_axis(dilate, 2, image, image.shape, [2, 2], core)
dilate_image_png = Image.fromarray(dilate_image)
dilate_image_png.save("dilate_image.png", "PNG")

# erode_image = erode(image, core, [2, 2], 1, 'white')
# erode_image_png = Image.fromarray(erode_image)
# erode_image_png.save("erode_image.png", "PNG")

print(f'{(time.time() - start_time)} seconds')
