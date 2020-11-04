import os
import os.path as osp
import gc
from datetime import datetime
import argparse
from typing import Tuple, List

import numpy as np
import tifffile as tif
import cv2 as cv
import dask

from slicer import split_image_into_blocks_of_size
from metadata_handling import get_cycle_composition
from warper import Warper
Image = np.ndarray


def convertu8(img: Image) -> Image:
    u8img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return u8img


def register_tiles(ref_img: np.ndarray, moving_img: np.ndarray) -> np.ndarray:
    ref_img_u8 = cv.normalize(ref_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    moving_img_u8 = cv.normalize(moving_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    return cv.calcOpticalFlowFarneback(moving_img_u8, ref_img_u8,
                                       None, pyr_scale=0.5, levels=3,
                                       winsize=21,
                                       iterations=3, poly_n=7, poly_sigma=1.3,
                                       flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN)


def reg_big_image(ref_img: Image, moving_img: Image, warper, block_width, block_height, overlap) -> Tuple[Image, List[np.ndarray]]:
    """ Calculates optical flow from moving_img to ref_img.
        Image is divided into pieces to decrease memory consumption.
        Currently working optical flow method is Farneback.
        Other methods either to complex to work with or don't have proper API for Python.
    """

    ref_img_tiles, ref_img_slice_info = split_image_into_blocks_of_size(ref_img, block_width, block_height, overlap)
    moving_img_tiles, moving_image_slice_info = split_image_into_blocks_of_size(moving_img, block_width, block_height, overlap)

    reg_task = []
    for t in range(0, len(ref_img_tiles)):
        reg_task.append(dask.delayed(register_tiles)(ref_img_tiles[t], moving_img_tiles[t]))

    print(datetime.now(), 'registering reference channel tiles')
    flow_tiles = dask.compute(*reg_task)
    flow_tiles = list(flow_tiles)

    del ref_img_tiles, ref_img_slice_info

    print(datetime.now(), 'warping reference channel tiles')
    warper.image_tiles = moving_img_tiles
    warper.slicer_info = moving_image_slice_info
    warper.flow_tiles = flow_tiles
    warped_moving_image = warper.warp()

    return warped_moving_image, flow_tiles


def channel_saving_first_cycle(writer, image, ref_position_in_cycle, cycle_size, cycle_number, in_path, meta):
    if ref_position_in_cycle != 0:
        for c in range(0, ref_position_in_cycle):
            key = cycle_number * cycle_size + c
            writer.save(tif.imread(in_path, key=key), photometric='minisblack', description=meta)

    writer.save(image, photometric='minisblack', description=meta)
    del image

    # if there are other channels after first ref channel warp and write them
    if ref_position_in_cycle != cycle_size - 1:
        for c in range(ref_position_in_cycle + 1, cycle_size):
            key = cycle_number * cycle_size + c
            writer.save(tif.imread(in_path, key=key), photometric='minisblack', description=meta)


def channel_saving(writer, warper, image, flow_tiles, ref_position_in_cycle, cycle_size, cycle_number, in_path, meta):
    if ref_position_in_cycle != 0:
        for c in range(0, ref_position_in_cycle):
            key = cycle_number * cycle_size + c
            warper.image = tif.imread(in_path, key=key)
            warper.flow_tiles = flow_tiles
            warped_image = warper.warp()
            writer.save(warped_image, photometric='minisblack', description=meta)

    writer.save(image, photometric='minisblack', description=meta)
    del image

    # if there are other channels after first ref channel warp and write them
    if ref_position_in_cycle != cycle_size - 1:
        for c in range(ref_position_in_cycle + 1, cycle_size):
            key = cycle_number * cycle_size + c
            warper.image = tif.imread(in_path, key=key)
            warper.flow_tiles = flow_tiles
            warped_image = warper.warp()
            writer.save(warped_image, photometric='minisblack', description=meta)


def register(in_path: str, out_dir: str, cycle_size: int, ncycles: int, ref_position_in_cycle: int, meta: str, warper, block_width, block_height, overlap):
    """ Read images and register them sequentially: 1<-2, 2<-3, 3<-4 etc.
        It is assumed that there is equal number of channels in each cycle.
    """
    filename = osp.basename(in_path).replace('.tif', '_opt_flow_registered.tif')
    out_path = osp.join(out_dir, filename)
    TW_img = tif.TiffWriter(out_path, bigtiff=True)

    first_ref_id = ref_position_in_cycle
    for i in range(0, ncycles-1):
        this_ref_id = cycle_size * i + ref_position_in_cycle
        next_ref_id = cycle_size * (i + 1) + ref_position_in_cycle
        print('\n{time} Processing cycle {this_cycle}/{total_cycles}'.format(time=str(datetime.now()), this_cycle=i+2, total_cycles=ncycles))
        
        # first reference channel processed separately from other
        if this_ref_id == first_ref_id:
            im1 = tif.imread(in_path, key=this_ref_id)
            im2 = tif.imread(in_path, key=next_ref_id)
            # register and warp 2nd ref image and get optical flow
            im2_warped, flow = reg_big_image(im1, im2, warper, block_width, block_height, overlap)
            del im2
            print(datetime.now(), 'warping and writing to file the rest of the channels')

            channel_saving_first_cycle(TW_img, im1, ref_position_in_cycle, cycle_size, i, in_path, meta)
            channel_saving(TW_img, warper, im2_warped, flow, ref_position_in_cycle, cycle_size, i+1, in_path, meta)

            del flow
            gc.collect()

        else:
            im1 = im2_warped  # this_ref_id # reuse warped image from previous cycle
            im2 = tif.imread(in_path, key=next_ref_id)
            im2_warped, flow = reg_big_image(im1, im2, warper, block_width, block_height, overlap)
            del im2
            print(datetime.now(), 'warping and writing to file the rest of the channels')
            channel_saving(TW_img, warper, im2_warped, flow, ref_position_in_cycle, cycle_size, i+1, in_path, meta)

            del flow
            gc.collect()

    TW_img.close()


def main(in_path: str, ref_channel: str, out_dir: str, n_workers: int, tile_size: int, overlap: int, ):

    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    if n_workers == 1:
        dask.config.set({'scheduler': 'synchronous'})
    else:
        dask.config.set({'num_workers': n_workers, 'scheduler': 'processes'})

    st = datetime.now()

    with tif.TiffFile(in_path, is_ome=True) as stack:
        ome = stack.ome_metadata

    cycle_size, ncycles, first_ref_position = get_cycle_composition(ome, ref_channel)
    block_width = tile_size
    block_height = tile_size
    overlap = overlap

    warper = Warper()
    warper.block_w = block_width
    warper.block_h = block_height
    warper.overlap = overlap



    # perform registration of full stack
    register(in_path, out_dir, cycle_size, ncycles, first_ref_position, ome, warper, block_width, block_height, overlap)

    fin = datetime.now()
    print('time elapsed', fin - st)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='image stack to register')
    parser.add_argument('-c', type=str, required=True, help='channel for registration')
    parser.add_argument('-o', type=str, required=True, help='output dir')
    parser.add_argument('-n', type=int, default=1, help='multiprocessing: number of processes, default 1')
    parser.add_argument('--tile_size', type=int, default=1000, help='size of a side of a square tile, ' +
                                                                    'e.g. --tile_size 1000 = tile with dims 1000x1000px')
    parser.add_argument('--overlap', type=int, default=50, help='size of the overlap for one side of the image,' +
                                                                'e.g. --overlap 50 = left,right,top,bottom overlaps are 50px each')

    args = parser.parse_args()

    main(args.i, args.c, args.o, args.n, args.tile_size, args.overlap)
