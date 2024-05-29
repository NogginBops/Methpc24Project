# Dependenices

Building this project is only tested on Ubuntu 20.04 but should with only minor tweaks run on other locations.

The paths to the hdf5 library comes from `h5cc -show`, to get `h5cc` on ubuntu it's easiest to install `libhdf5-dev` which contains it.

The HDF5 library can be downloaded from: https://www.hdfgroup.org/downloads/hdf5 and can be built using their build instructions.
To use your own build of HDF5 you need to replace the flags in the make file with your own `h5cc -show` result.

To build this project the `libhdf5-dev` package needs to be installed.

## Building

To build, simply run `make`.

# Running

Run the build binary as such `mpirun -np <processes> ./volume_render` with `<processes>` replaced with the desired number of processes.

The output images are going to be written to `./results/`.