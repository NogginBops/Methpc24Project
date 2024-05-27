#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>

#include <hdf5/serial/hdf5.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef RES_X
#define RES_X 250
#endif
#ifndef RES_Y
#define RES_Y 250
#endif
#ifndef RES_Z
#define RES_Z 250
#endif

#define SCALE 0.65f

#define GRID_AT(grid, dims, x, y, z) (grid)[(z) * (dims)[0] * (dims)[1] + (y) * (dims)[0] + (x)]

#define LERP(a, b, t) (1 - (t)) * (a) + (t) * (b)

#define CLAMP(x, min, max) (x) < (min) ? (min) : (x) > (max) ? (max) : (x)

typedef struct {
    float r, g, b, a;
} rgba_32f;

typedef struct {
    uint8_t r, g, b, a;
} rgba_8i;

float *read_hdf5_data(char *filepath, char *dataset_name, hsize_t dims[3]) {
    hid_t file = H5Fopen(filepath, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(file != H5I_INVALID_HID);

    hsize_t size;
    herr_t err = H5Fget_filesize(file, &size);
    assert(err >= 0);
    printf("File size: %llu bytes\n", size);

    hid_t dataset = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    assert(dataset != H5I_INVALID_HID);

    hsize_t storage_size = H5Dget_storage_size(dataset);
    printf("storage_size=%llu\n", storage_size);
    
    hid_t datatype = H5Dget_type(dataset);
    assert(datatype != H5I_INVALID_HID);

    size_t datatype_size = H5Tget_size(datatype);
    printf("datatype_size=%lu\n", datatype_size);
    assert(datatype_size == 4);

    H5T_class_t datatype_class = H5Tget_class(datatype);
    switch (datatype_class)
    {
        case H5T_NO_CLASS:  printf("No class\n");  break;
        case H5T_INTEGER:   printf("Integer\n");   break;
        case H5T_FLOAT:     printf("Float\n");     break;
        case H5T_TIME:      printf("Time\n");      break;
        case H5T_STRING:    printf("String\n");    break;
        case H5T_BITFIELD:  printf("Bitfield\n");  break;
        case H5T_OPAQUE:    printf("Opaque\n");    break;
        case H5T_COMPOUND:  printf("Compound\n");  break;
        case H5T_REFERENCE: printf("Reference\n"); break;
        case H5T_ENUM:      printf("Enum\n");      break;
        case H5T_VLEN:      printf("Vlen\n");      break;
        case H5T_ARRAY:     printf("Array\n");     break;
        default:            printf("Unknown\n");   break;
    }
    assert(datatype_class == H5T_FLOAT);

    hid_t space = H5Dget_space(dataset);
    assert(space != H5I_INVALID_HID);

    int rank = H5Sget_simple_extent_ndims(space);
    assert(rank == 3);

    int rank2 = H5Sget_simple_extent_dims(space, dims, NULL);
    assert(rank2 == rank);

    float* data = malloc(storage_size);
    assert(H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0);

    printf("Read data %id (%llix%llix%lli)\n", rank, dims[0], dims[1], dims[2]);

    H5Dclose(dataset);
    H5Fclose(file);

    return data;
}

// Is there a faster way to do tri-linear interpolation?
float interpolate_grid(float* data, hsize_t dims[3], float x, float y, float z) {
    if (x < 0 || x >= dims[0]-1) return 0;
    if (y < 0 || y >= dims[1]-1) return 0;
    if (z < 0 || z >= dims[2]-1) return 0;

    int ix = x;
    int iy = y;
    int iz = z;

    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    
    float f000 = GRID_AT(data, dims, ix + 0, iy + 0, iz + 0);
    float f001 = GRID_AT(data, dims, ix + 0, iy + 0, iz + 1);
    float f010 = GRID_AT(data, dims, ix + 0, iy + 1, iz + 0);
    float f011 = GRID_AT(data, dims, ix + 0, iy + 1, iz + 1);
    float f100 = GRID_AT(data, dims, ix + 1, iy + 0, iz + 0);
    float f101 = GRID_AT(data, dims, ix + 1, iy + 0, iz + 1);
    float f110 = GRID_AT(data, dims, ix + 1, iy + 1, iz + 0);
    float f111 = GRID_AT(data, dims, ix + 1, iy + 1, iz + 1);
    
    float fx00 = LERP(f000, f100, fx);
    float fx01 = LERP(f001, f101, fx);
    float fx10 = LERP(f010, f110, fx);
    float fx11 = LERP(f011, f111, fx);

    float fxy0 = LERP(fx00, fx10, fy);
    float fxy1 = LERP(fx01, fx11, fy);

    float fxyz = LERP(fxy0, fxy1, fz);

    return fxyz;
}

rgba_32f transfer_function(float x) {
    return (rgba_32f) {
        .r = 1.0f * expf( -powf(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * expf( -powf(x - 3.0f, 2.0f) / 0.1f ) + 0.1f  * expf( -powf(x - -3.0f, 2.0f) / 0.5f ),
        .g = 1.0f * expf( -powf(x - 9.0f, 2.0f) / 1.0f ) + 1.0f * expf( -powf(x - 3.0f, 2.0f) / 0.1f ) + 0.1f  * expf( -powf(x - -3.0f, 2.0f) / 0.5f ),
        .b = 0.1f * expf( -powf(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * expf( -powf(x - 3.0f, 2.0f) / 0.1f ) + 1.0f  * expf( -powf(x - -3.0f, 2.0f) / 0.5f ),
        .a = 0.6f * expf( -powf(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * expf( -powf(x - 3.0f, 2.0f) / 0.1f ) + 0.01f * expf( -powf(x - -3.0f, 2.0f) / 0.5f ),
    };
}

int main() {
    printf("Hello, world!\n");

    int ret_val = 0;
    assert(H5open() >= 0);

    {
        unsigned int majnum, minnum, relnum;
        hbool_t flag;

        assert(H5get_libversion(&majnum, &minnum, &relnum) >= 0);

        assert(H5is_library_threadsafe(&flag) >= 0);

        printf("Welcome to HDF5 %d.%d.%d\n", majnum, minnum, relnum);
        printf("Thread-safety %s\n", (flag > 0) ? "enabled" : "disabled");
    }

    {
        hsize_t dims[3];
        float* data = read_hdf5_data("../data/datacube.hdf5", "density", dims);

        interpolate_grid(data, dims, 0.1f, 0.5f, 0.0f);

        rgba_8i* image = calloc(RES_X * RES_Y, sizeof(rgba_8i));

        const int NAngles = 24;
        for (int iangle = 0; iangle < NAngles; iangle++) {
            float angle = 2 * M_PI * (iangle / (float)NAngles);

            // now I have the datacube!
            #pragma omp parallel for
            for (int ix = 0; ix < RES_X; ix++) {
                float x = ((ix / (RES_X - 1.0f)) - 0.5f) * (dims[0] * SCALE);
                for (int iy = 0; iy < RES_Y; iy++) {
                    float y = ((iy / (RES_Y - 1.0f)) - 0.5f) * (dims[1] * SCALE);

                    rgba_32f sum = {0};
                    for (int iz = RES_Z - 1; iz >= 0; iz--) {
                        float z = ((iz / (RES_Z - 1.0f)) - 0.5f) * (dims[2] * SCALE);

                        float fx = x * cosf(angle) - z * sinf(angle);
                        float fy = y;
                        float fz = x * sinf(angle) + z * cosf(angle);

                        // Center the datacube.
                        fx += (dims[0]/2.0f);
                        fy += (dims[1]/2.0f);
                        fz += (dims[2]/2.0f);

                        float density = interpolate_grid(data, dims, fx, fy, fz);

                        rgba_32f c = transfer_function(logf(density));

                        sum.r = c.a * c.r + (1 - c.a) * sum.r;
                        sum.g = c.a * c.g + (1 - c.a) * sum.g;
                        sum.b = c.a * c.b + (1 - c.a) * sum.b;
                    }

                    image[iy * RES_X + ix] = (rgba_8i) {
                        .r = CLAMP(sum.r * 255, 0, 255),
                        .g = CLAMP(sum.g * 255, 0, 255),
                        .b = CLAMP(sum.b * 255, 0, 255),
                        .a = 255,
                    };
                }
            }

            mkdir("results", 0777);

            char name[256];
            snprintf(name, 256, "./results/result%i.png", iangle);
            int res = stbi_write_png(name, RES_X, RES_Y, 4, image, RES_X * sizeof(rgba_8i));
            printf("Wrote %s\n", name);
            assert(res != 0);
        }
    }

    assert(H5close() >= 0);
    return ret_val;
}

