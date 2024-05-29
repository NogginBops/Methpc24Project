.PHONY = launch clean

BINARY = volume_render

.DEFAULT_GOAL := $(BINARY)

SRCS := main.c

CC := h5cc
CCFlags := -o $(BINARY)
CCFlags := $(CCFlags)
CCFlags := $(CCFlags) -Wall -W
CCFlags := $(CCFlags) -lm -lz -lOpenCL
# FIXME: Only add this for debug?
CCFlags := $(CCFlags) -g

# From h5cc -show
CCFlags := $(CCFlags) -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a -lcrypto -lcurl -lpthread -lsz -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/serial

all: $(BINARY)

$(BINARY): $(SRCS)
	$(CC) $(CCFlags) $(ARG_FLAGS) $(SRCS)

run: $(BINARY)
	$(BINARY)

clean:
	@rm $(BINARY)