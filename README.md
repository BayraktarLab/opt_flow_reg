# Optical flow based registration

Currently limited to stack of maxz projections.

## Command line arguments

**`-i`**  path to image stack

**`-c`**  name of reference channel

**`-o`**  output directory

**`-n`**  multiprocessing: number of processes, default 2

## Example usage

**`python opt_flow_reg.py -i /path/to/iamge/stack/out.tif -c "Atto 490LS" -o /path/to/output/dir/ -n 3`**


## Dependencies
numpy tifffile opencv-contrib-python "dask[delayed]"