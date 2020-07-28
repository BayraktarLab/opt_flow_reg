from typing import List, Tuple, Union

import numpy as np

Image = np.ndarray


def stitch_image(img_list: List[Image], slicer_info: dict) -> Image:

    x_nblocks = slicer_info['nblocks']['x']
    y_nblocks = slicer_info['nblocks']['y']
    block_shape = slicer_info['block_shape']
    overlap = slicer_info['overlap']
    padding = slicer_info['padding']

    x_axis = -1
    y_axis = -2

    block_x_size = block_shape[x_axis]
    block_y_size = block_shape[y_axis]

    padding_left = padding["left"]
    padding_right = padding["right"]
    padding_top = padding["top"]
    padding_bottom = padding["bottom"]

    big_image_x_size = (x_nblocks * block_x_size) - padding_left - padding_right
    big_image_y_size = (y_nblocks * block_y_size) - padding_top - padding_bottom
    dtype = img_list[0].dtype
    big_image_shape = (big_image_y_size, big_image_x_size)
    big_image = np.zeros(big_image_shape, dtype=dtype)

    big_image_slice = [slice(None), slice(None)]
    block_slice = [slice(None), slice(None)]

    n = 0
    for i in range(0, y_nblocks):
        yf = i * block_y_size
        yt = yf + block_y_size

        if i == 0:
            block_slice[y_axis] = slice(0 + overlap + padding_top, block_y_size + overlap)
            big_image_slice[y_axis] = slice(padding_top, yt)
        elif i == y_nblocks - 1:
            block_slice[y_axis] = slice(0 + overlap, block_y_size + overlap - padding_bottom)
            big_image_slice[y_axis] = slice(yf, yt - padding_bottom)
        else:
            block_slice[y_axis] = slice(0 + overlap, block_y_size + overlap)
            big_image_slice[y_axis] = slice(yf, yt)

        for j in range(0, x_nblocks):
            xf = j * block_x_size
            xt = xf + block_x_size

            if j == 0:
                block_slice[x_axis] = slice(0 + overlap + padding_left, block_x_size + overlap)
                big_image_slice[x_axis] = slice(padding_left, xt)
            elif j == x_nblocks - 1:
                block_slice[x_axis] = slice(0 + overlap, block_x_size + overlap - padding_right)
                big_image_slice[x_axis] = slice(xf, xt - padding_right)
            else:
                block_slice[x_axis] = slice(0 + overlap, block_x_size + overlap)
                big_image_slice[x_axis] = slice(xf, xt)

            block = img_list[n]

            big_image[tuple(big_image_slice)] = block[tuple(block_slice)]

            n += 1

    return big_image
