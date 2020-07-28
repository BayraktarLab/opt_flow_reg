import copy

import numpy as np

Image = np.ndarray


def split_image_into_blocks_of_size(arr: Image, block_w: int, block_h: int, overlap: int):
    """ Splits image into blocks of given size.
        block_w - block width
        block_h - block height
    """
    img_width, img_height = arr.shape[-1], arr.shape[-2]
    x_axis = -1
    y_axis = -2
    arr_shape = list(arr.shape)
    dtype = arr.dtype

    x_nblocks = arr_shape[x_axis] // block_w if arr_shape[x_axis] % block_w == 0 else (arr_shape[x_axis] // block_w) + 1
    y_nblocks = arr_shape[y_axis] // block_h if arr_shape[y_axis] % block_h == 0 else (arr_shape[y_axis] // block_h) + 1

    # check if image size is divisible by block size
    pad_shape = copy.copy(arr_shape)
    if img_height % block_h != 0:
        do_horizontal_padding = True
    else:
        do_horizontal_padding = False

    if img_width % block_w != 0:
        do_vertical_padding = True
    else:
        do_vertical_padding = False

    blocks = []

    # row
    for i in range(0, y_nblocks):
        # height of this block
        ver_f = block_h * i
        ver_t = ver_f + block_h

        if overlap != 0:
            # vertical overlap of this block
            if i == 0:
                ver_t += overlap
            elif i == y_nblocks - 1:
                ver_f -= overlap
                ver_t = img_height
            else:
                ver_f -= overlap
                ver_t += overlap

        # col
        for j in range(0, x_nblocks):

            # width of this block
            hor_f = block_w * j
            hor_t = hor_f + block_w

            if overlap != 0:
                # horizontal overlap of this block
                if j == 0:
                    hor_t += overlap
                elif j == x_nblocks - 1:
                    hor_f -= overlap
                    hor_t = img_width
                else:
                    hor_f -= overlap
                    hor_t += overlap

            block = arr[ver_f: ver_t, hor_f: hor_t]

            # handling cases when image size is not divisible by block size
            if j == x_nblocks - 1 and do_horizontal_padding:
                pad_shape = list(block.shape)
                rest = img_width % block_w
                pad_shape[x_axis] = block_w - rest
                pad_shape[y_axis] = block.shape[y_axis]  # width of padding
                block = np.concatenate((block, np.zeros(pad_shape, dtype=dtype)), axis=x_axis)

            if i == y_nblocks - 1 and do_vertical_padding:
                pad_shape = list(block.shape)
                rest = img_height % block_h
                pad_shape[x_axis] = block.shape[x_axis]
                pad_shape[y_axis] = block_h - rest  # height of padding
                block = np.concatenate((block, np.zeros(pad_shape, dtype=dtype)), axis=y_axis)

            # handling cases when of overlap on the edge images
            if overlap != 0:
                overlap_pad_shape = list(block.shape)
                if i == 0:
                    overlap_pad_shape[x_axis] = block.shape[x_axis]
                    overlap_pad_shape[y_axis] = overlap
                    block = np.concatenate((np.zeros(overlap_pad_shape, dtype=dtype), block), axis=0)
                    # print('i0', block.shape)
                elif i == y_nblocks - 1:
                    overlap_pad_shape[x_axis] = block.shape[x_axis]
                    overlap_pad_shape[y_axis] = overlap
                    block = np.concatenate((block, np.zeros(overlap_pad_shape, dtype=dtype)), axis=0)
                    # print('i-1', block.shape)
                if j == 0:
                    overlap_pad_shape[x_axis] = overlap
                    overlap_pad_shape[y_axis] = block.shape[y_axis]
                    block = np.concatenate((np.zeros(overlap_pad_shape, dtype=dtype), block), axis=1)
                    # print('j0', block.shape)
                elif j == x_nblocks - 1:
                    overlap_pad_shape[x_axis] = overlap
                    overlap_pad_shape[y_axis] = block.shape[y_axis]
                    block = np.concatenate((block, np.zeros(overlap_pad_shape, dtype=dtype)), axis=1)
                    # print('j-1', block.shape)
            # print('\n')
            blocks.append(block)

    block_shape = [block_h, block_w]
    nblocks = dict(x=x_nblocks, y=y_nblocks)
    padding = dict(left=0, right=0, top=0, bottom=0)
    padding["right"] = block_w - (img_width % block_w)
    padding["bottom"] = block_h - (img_height % block_h)

    info = dict(block_shape=block_shape, nblocks=nblocks, overlap=overlap, padding=padding)

    return blocks, info


def split_by_nblocks(arr: Image, x_nblocks: int, y_nblocks: int, overlap: int):
    """ Splits image into blocks by number of block.
        x_nblocks - number of blocks horizontally
        y_nblocks - number of blocks vertically
    """
    img_width, img_height = arr.shape[-1], arr.shape[-2]
    block_w = img_width // x_nblocks
    block_h = img_height // y_nblocks
    return split_image_into_blocks_of_size(arr, block_w, block_h, overlap)
