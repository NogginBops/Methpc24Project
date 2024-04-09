.DEFAULT_GOAL := compile

BINARY = volume_render

HDF5_INCLUDE_DIR := ../hdf5-1.14.3/hdf5/include/
HDF5_STATIC_LIB := ../hdf5-1.14.3/hdf5/lib/libhdf5.a ../hdf5-1.14.3/hdf5/lib/libhdf5_hl.a

SRCS := main.c ${HDF5_STATIC_LIB}

CC := cc
CCFlags := -o $(BINARY)
CCFlags := $(CCFlags) -O3
CCFlags := $(CCFlags) -Wall -W
CCFlags := $(CCFlags) -lm -lz
CCFlags := $(CCFlags) -I$(HDF5_INCLUDE_DIR)
# FIXME: Only add this for debug?
CCFlags := $(CCFlags) -g

compile: $(SRCS)
	$(CC) $(SRCS) $(CCFlags)

clean:
	@rm $(BINARY)