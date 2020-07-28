## Optical flow based registration for immunofluorescence images

These scripts perform fine registration using warping. 
A map for warping is calculated using Farneback optical flow algorithm.
This method is sensitive to pixel intensities, so images must have similar scale of pixel intensities. 
For better results choose signal channel that is present in all imaging cycles, but not DAPI. 

Currently does not support z-stacks.

### Command line arguments

**`-i`**  path to image stack

**`-c`**  name of reference channel

**`-o`**  output directory

**`-n`**  multiprocessing: number of processes, default 1

### Example usage

**`python opt_flow_reg.py -i /path/to/iamge/stack/out.tif -c "Atto 490LS" -o /path/to/output/dir/ -n 3`**


### Dependencies
`numpy tifffile opencv-contrib-python dask`

