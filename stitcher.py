from typing import List, Tuple, Union

import numpy as np

Image = np.ndarray


def get_slices(arr: np.ndarray, hor_f: int, hor_t: int, ver_f: int, ver_t: int, padding: dict, overlap=0):
    left_check  = hor_f - padding['left']
    top_check   = ver_f - padding['top']
    right_check = hor_t - arr.shape[-1]
    bot_check   = ver_t - arr.shape[-2]

    left_pad_size = 0
    top_pad_size = 0
    right_pad_size = 0
    bot_pad_size = 0

    if left_check < 0:
        left_pad_size = abs(left_check)
        hor_f = 0
    if top_check < 0:
        top_pad_size = abs(top_check)
        ver_f = 0
    if right_check > 0:
        right_pad_size = right_check
        hor_t = arr.shape[1]
    if bot_check > 0:
        ver_t = arr.shape[0]

    big_image_slice = (slice(ver_f, ver_t), slice(hor_f, hor_t))
    block_shape = (ver_t - ver_f, hor_t - hor_f)
    block_slice = (slice(top_pad_size + overlap, block_shape[0] + overlap), slice(left_pad_size + overlap, block_shape[1] + overlap))

    return big_image_slice, block_slice


def stitch_image(img_list: List[Image], slicer_info) -> Image:
    x_nblocks = slicer_info['nblocks']['x']
    y_nblocks = slicer_info['nblocks']['y']
    block_shape = slicer_info['block_shape']
    overlap = slicer_info['overlap']
    padding = slicer_info['padding']

    x_axis = -1
    y_axis = -2

    block_x_size = block_shape[x_axis] - overlap * 2
    block_y_size = block_shape[y_axis] - overlap * 2

    big_image_x_size = (x_nblocks * block_x_size) - padding["left"] - padding["right"]
    big_image_y_size = (y_nblocks * block_y_size) - padding["top"] - padding["bottom"]

    big_image_shape = (big_image_y_size, big_image_x_size)
    dtype = img_list[0].dtype
    big_image = np.zeros(big_image_shape, dtype=dtype)

    n = 0
    for i in range(0, y_nblocks):
        ver_f = i * block_y_size
        ver_t = ver_f + block_y_size

        for j in range(0, x_nblocks):
            hor_f = j * block_x_size
            hor_t = hor_f + block_x_size

            big_image_slice, block_slice = get_slices(big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap)
            block = img_list[n]

            big_image[tuple(big_image_slice)] = block[tuple(block_slice)]

            n += 1

    return big_image

