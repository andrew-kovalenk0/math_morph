from PIL import Image
import numpy as np


def dilate(image, struct, point, step):
    result_image = image.copy()
    for i in range(0, image.shape[0], step):
        for j in range(0, image.shape[1], step):
            if (image[i][j] == [255, 255, 255]).all():
                for ind_x, x in enumerate(range(i - point[0], i + struct.shape[0] - point[0])):
                    for ind_y, y in enumerate(range(j - point[1], j + struct.shape[1] - point[1])):
                        if (struct[ind_x][ind_y] == [255, 255, 255]).all():
                            result_image[x][y] = [255, 255, 255]
    return result_image


image_png = Image.open('image.png')
array_image = np.asarray(image_png)
struct_png = Image.open('struct.png')
struct_element = np.asarray(struct_png)

dilation_image = dilate(array_image, struct_element, [1, 1], 1)

image2 = Image.fromarray(dilation_image)
image2.save("image2.png", "PNG")
