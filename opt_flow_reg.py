import os
import re
import gc
from datetime import datetime
import argparse
from typing import Tuple, List

import numpy as np
import tifffile as tif
import cv2 as cv
import dask

Image = np.ndarray


def draw_hsv(flow: np.ndarray) -> Image:
    """ Can be used to visualize optical flow """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv.normalize(v, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img: Image, flow: np.ndarray) -> np.ndarray:
    """ Warps input image according to optical flow """
    h, w = flow.shape[:2]
    xflow = -flow
    xflow[:, :, 0] += np.arange(w)
    xflow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, xflow, None, cv.INTER_LINEAR)
    return res


def convertu8(img: Image) -> Image:
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def register_pieces(ref_img: np.ndarray, moving_img: np.ndarray, f: int, t: int) -> np.ndarray:
    return cv.calcOpticalFlowFarneback(convertu8(moving_img[f:t, :]), convertu8(ref_img[f:t, :]),
                                       None, pyr_scale=0.6, levels=5,
                                       winsize=21,
                                       iterations=3, poly_n=7, poly_sigma=1.3,
                                       flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)


def warp_pieces(moving_img: np.ndarray, flow: np.ndarray, f: int, t: int, i: int) -> Image:
    print(i)
    return warp_flow(moving_img[f:t, :], flow[f:t, :, :])


def assemble_from_pieces(pieces: List[Image], overlap: int) -> Image:
    #total_overlap_size = overlap * (len(pieces) - 2)

    x_shapes = []
    y_shapes = []
    for im in pieces:
        y_shape, x_shape = im.shape
        y_shapes.append(y_shape - overlap)
        x_shapes.append(x_shape)

    y_shapes[0] += overlap

    y_pos = np.cumsum(y_shapes)
    y_pos.insert(0, 0)

    y_size = sum(y_shapes)
    x_size = x_shapes[0]

    big_image = np.zeros((y_size, x_size), dtype=pieces[0].dtype)

    for i in range(0, len(pieces) - 1):
        this_image = pieces[i]
        next_image = pieces[i + 1]

        f = y_pos[i]
        t = y_pos[i + 1]

        this_image[-overlap:, :] = np.mean((this_image[-overlap:, :], next_image[:overlap, :]), axis=0)
        pieces[i + 1] = next_image[overlap:, :]
        big_image[f:t, :] = this_image

    big_image[y_pos[-1]:-1:, :] = pieces[-1]

    return big_image


def reg_big_image(ref_img: Image, moving_img: Image) -> Tuple[Image, np.ndarray]:
    """ Calculates optical flow from moving_img to ref_img.
        Image is divided into pieces to decrease memory consumption.
        Currently working optical flow method is Farneback.
        Other methods either to complex to work with or don't have proper API for Python.
    """
    n_pieces = 10
    overlap = 20
    row_pieces = ref_img.shape[0] // n_pieces
    reg_task = []
    delayed_ref = dask.delayed(ref_img)
    delayed_mov = dask.delayed(moving_img)
    for i in range(0, n_pieces):
        #print(i)
        if i == 0:
            f = 0
            t = row_pieces + overlap
        elif i == n_pieces - 1:
            f = (i * row_pieces) - overlap
            t = -1
        else:
            f = (i * row_pieces) - overlap  # from
            t = (f + row_pieces) + overlap  # to

        reg_task.append(dask.delayed(register_pieces)(delayed_ref, delayed_mov, f, t))
    print('registering pieces')
    flow_li = dask.compute(*reg_task)
    #flow_assembled = np.concatenate(flow_li, axis=0)
    flow_assembled = assemble_from_pieces(flow_li, 20)
    del flow_li, reg_task
    gc.collect()

    warp_task = []
    #delayed_flow = dask.delayed(flow_assembled)
    for i in range(0, n_pieces):
        # print(i)
        f = i * row_pieces  # from
        t = f + row_pieces  # to
        if i == n_pieces - 1:
            t = ref_img.shape[0]

        warp_task.append(dask.delayed(warp_pieces)(moving_img, flow_assembled, f, t, i))

    print('warping pieces')
    warp_li = dask.compute(*warp_task)
    del delayed_mov, delayed_ref
    img_assembled = np.concatenate(warp_li, axis=0)

    return img_assembled, flow_assembled


def register(in_path: str, out_path: str, channels: list, meta: str):
    """ Read images and register them sequentially: 1<-2, 2<-3, 3<-4 etc. """
    filename = os.path.basename(in_path).replace('.tif', '_opt_flow.tif')

    ref_ch_ids = [i for i, c in enumerate(channels) if c == 1]
    first_ref = ref_ch_ids[0]

    TW_img = tif.TiffWriter(out_path + filename, bigtiff=True)
    for i in range(0, len(ref_ch_ids) - 1):
        this_ref = ref_ch_ids[i]
        next_ref = ref_ch_ids[i + 1]
        print('Processing cycle {0}/{1}'.format(i+1, len(ref_ch_ids)))
        
        # first reference channel processed separately from other
        if this_ref == first_ref:
            im1 = tif.imread(in_path, key=this_ref)
            im2 = tif.imread(in_path, key=next_ref)
            # register and warp 2nd ref image and get optical flow
            im2_warped, flow = reg_big_image(im1, im2, method='farneback')
            del im2

            # collect images before first reference image if they exist
            ch_before_first_ref = []
            if first_ref == 0:
                ch_before_first_ref = None
            else:
                for c in range(0, first_ref):
                    ch_before_first_ref.append(tif.imread(in_path, key=c))

            # warp other images before second reference image if they exist
            ch_before_second_ref = []
            if first_ref + 1 == next_ref:
                ch_before_second_ref = None
            else:
                for c in range(first_ref + 1, next_ref):
                    ch_before_second_ref.append(warp_flow(tif.imread(in_path, key=c), flow))

            # write images on disk
            if ch_before_first_ref is None and ch_before_second_ref is None:
                TW_img.save(np.stack((im1, im2_warped), axis=0), photometric='minisblack', description=meta)
            elif ch_before_first_ref is not None and ch_before_second_ref is None:
                TW_img.save(np.stack((*ch_before_first_ref, im1, im2_warped), axis=0), photometric='minisblack', description=meta)
            elif ch_before_first_ref is None and ch_before_second_ref is not None:
                TW_img.save(np.stack((im1, *ch_before_second_ref, im2_warped), axis=0), photometric='minisblack', description=meta)
            else:
                TW_img.save(np.stack((*ch_before_first_ref, im1, *ch_before_second_ref, im2_warped), axis=0), photometric='minisblack', description=meta)

            gc.collect()
            # TW_flow.save(flow[:,:,:], photometric='minisblack')
        else:
            im1 = im2_warped  # this_ref # reuse warped image from previous cycle
            im2 = tif.imread(in_path, key=next_ref)
            im2_warped, flow = reg_big_image(im1, im2, method='farneback')

            # warp other images before reference image if they exist
            ch_before_next_ref = []
            if this_ref + 1 == next_ref:
                ch_before_next_ref = None
            else:
                for c in range(this_ref + 1, next_ref):
                    ch_before_next_ref.append(warp_flow(tif.imread(in_path, key=c), flow))

            # write images on disk
            if ch_before_next_ref is None:
                TW_img.save(im2_warped, photometric='minisblack', description=meta)
            else:
                TW_img.save(np.stack((*ch_before_next_ref, im2_warped), axis=0), photometric='minisblack', description=meta)

            del im2
            gc.collect()
            # TW_flow.save(flow[:,:,:], photometric='minisblack')

    TW_img.close()
    # TW_flow.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='image stack to register')
    parser.add_argument('-c', type=str, required=True, help='channel for registration')
    parser.add_argument('-o', type=str, required=True, help='output dir')
    parser.add_argument('-n', type=int, default=1, help='multiprocessing: number of processes, default 1')
    args = parser.parse_args()

    in_path = args.i
    ref_channel = args.c
    out_path = args.o
    n_workers = args.n

    if not out_path.endswith('/'):
        out_path += '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if n_workers == 1:
        dask.config.set({'scheduler': 'synchronous'})
    else:
        dask.config.set({'num_workers': n_workers, 'scheduler': 'processes'})

    st = datetime.now()

    stack = tif.TiffFile(in_path, is_ome=True)
    ome = stack.ome_metadata
    stack.close()

    # find selected reference channels in ome metadata
    matches = re.findall(r'Fluor=".*?"', ome)
    matches = [m.replace('Fluor=', '').replace('"', '') for m in matches]
    matches = [re.sub(r'c\d+\s+', '', m) for m in matches]  # remove cycle name

    # encode reference channels as 1 other 0
    channels = []
    for i, channel in enumerate(matches):
        if channel == ref_channel:
            channels.append(1)
        else:
            channels.append(0)
    
    # check if reference channel is available
    if ref_channel not in matches:
        raise ValueError('Incorrect reference channel. Available reference channels ' + ', '.join(set(matches)))

    # perform registration of full stack
    register(in_path, out_path, channels, ome)

    fin = datetime.now()
    print('time elapsed', fin - st)


if __name__ == '__main__':
    main()
