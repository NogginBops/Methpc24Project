#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>

#include <hdf5/serial/hdf5.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

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

const char *clGetErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");

    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);  /* same as rewind(f); */

    char *string = malloc(fsize + 1);
    fread(string, fsize, 1, file);
    fclose(file);

    string[fsize] = 0;
    return string;
}

int main() {
    int ret_val = 0;
    assert(H5open() >= 0);

    cl_platform_id platform_id;
    cl_uint platform_count;
    clGetPlatformIDs(1, &platform_id, &platform_count);
    if (platform_count == 0) {
        printf("No OpenCL platform available!\n");
        return EXIT_FAILURE;
    }

    char name[1024] = {0};
    size_t ret_num;
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 1024, name, &ret_num);
    printf("OpenCL platform name: %s\n", name);

    cl_device_id device_id;
    cl_uint device_count;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &device_count);
    if (device_count == 0) {
        printf("No OpenCL devices available!\n");
        return EXIT_FAILURE;
    }

    memset(name, 0, 1024);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, name, &ret_num);
    printf("OpenCL device name: %s\n", name);

    cl_int error = 0;
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);
    if (error != 0) {
        printf("Error creating OpenCL context: %s\n", clGetErrorString(error));
    }

    {
        unsigned int majnum, minnum, relnum;
        hbool_t flag;

        assert(H5get_libversion(&majnum, &minnum, &relnum) >= 0);

        assert(H5is_library_threadsafe(&flag) >= 0);

        //printf("Welcome to HDF5 %d.%d.%d\n", majnum, minnum, relnum);
        //printf("Thread-safety %s\n", (flag > 0) ? "enabled" : "disabled");
    }

    {
        hsize_t dims[3];
        float* data = read_hdf5_data("../data/datacube.hdf5", "density", dims);
        dims[0] = 256;
        dims[1] = 256;
        dims[2] = 256;

        cl_image_format img_format;
        img_format.image_channel_order = CL_R;
        img_format.image_channel_data_type = CL_FLOAT;
        cl_image_desc img_desc;
        img_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
        img_desc.image_width  = dims[0];
        img_desc.image_height = dims[1];
        img_desc.image_depth  = dims[2];
        img_desc.image_array_size = 0;
        img_desc.image_row_pitch = dims[0] * sizeof(float);
        img_desc.image_slice_pitch = dims[0] * dims[1] * sizeof(float);
        img_desc.num_mip_levels = 0;
        img_desc.num_samples = 0;
        img_desc.mem_object = NULL;

        cl_int error;
        cl_mem volume_image = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, &img_desc, data, &error);
        if (error != 0) {
            printf("clCreateImage volume error: %s\n", clGetErrorString(error));
        }

        cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
        if (error != 0) {
            printf("clCreateCommandQueueWithProperties error: %s\n", clGetErrorString(error));
        }

        const char* source_code = read_file("kernel.cl");
        cl_program program = clCreateProgramWithSource(context, 1, &source_code, NULL, &error);
        if (error != 0) {
            printf("clCreateProgramWithSource error: %s\n", clGetErrorString(error));
        }

        error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (error != 0) {
            printf("clBuildProgram error: %s\n", clGetErrorString(error));
        }

        char build_log[4096] = {0};
        size_t build_log_length;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 4096, build_log, &build_log_length);
        printf("Build log: %s\n", build_log);

        cl_kernel kernel = clCreateKernel(program, "render", &error);
        if (error != 0) {
            printf("clCreateKernel error: %s\n", clGetErrorString(error));
        }

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &volume_image);
        if (error != 0) {
            printf("clSetKernelArg error: %s\n", clGetErrorString(error));
        }

        const int ress[] = { RES_X, RES_Y, RES_Z, 0 };
        error = clSetKernelArg(kernel, 3, 4*sizeof(int), ress);
        if (error != 0) {
            printf("clSetKernelArg error: %s\n", clGetErrorString(error));
        }

        const int dimss[] = { dims[0], dims[1], dims[2], 0 };
        error = clSetKernelArg(kernel, 4, 4*sizeof(int), dimss);
        if (error != 0) {
            printf("clSetKernelArg error: %s\n", clGetErrorString(error));
        }

        printf("Starting render.\n");

        rgba_8i* image = calloc(RES_X * RES_Y, sizeof(rgba_8i));
        
        const int NAngles = 1;
        cl_mem result_images[NAngles];
        for (int iangle = 0; iangle < NAngles; iangle++) {
            float angle = 2 * M_PI * (iangle / (float)NAngles);

            img_format.image_channel_order = CL_RGBA;
            img_format.image_channel_data_type = CL_UNORM_INT8;

            img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            img_desc.image_width = RES_X;
            img_desc.image_height = RES_Y;
            img_desc.image_depth = 1;
            img_desc.image_array_size = 0;
            img_desc.image_row_pitch = 0;
            img_desc.image_slice_pitch = 0;
            img_desc.num_mip_levels = 0;
            img_desc.num_samples = 0;
            img_desc.mem_object = NULL;

            result_images[iangle] = clCreateImage(context, CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, &img_format, &img_desc, NULL, &error);
            if (error != 0) {
                printf("clCreateImage image error: %s\n", clGetErrorString(error));
            }

            error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_images[iangle]);
            if (error != 0) {
                printf("clSetKernelArg error: %s\n", clGetErrorString(error));
            }

            error = clSetKernelArg(kernel, 2, sizeof(float), &angle);
            if (error != 0) {
                printf("clSetKernelArg error: %s\n", clGetErrorString(error));
            }

            const size_t global_work_size[] = { RES_X, RES_Y };
            // FIXME: Better local worksize.
            const size_t local_work_size[] = { 4, 4 };
            error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
            if (error != 0) {
                printf("clEnqueueNDRangeKernel error: %s\n", clGetErrorString(error));
            }
        }

        for (int iangle = 0; iangle < NAngles; iangle++) {
            size_t origin[] = { 0, 0, 0 };
            size_t region[] = { RES_X, RES_Y, 1 };
            error = clEnqueueReadImage(command_queue, result_images[iangle], CL_TRUE, origin, region, 0, 0, image, 0, NULL, NULL);
            if (error != 0) {
                printf("clEnqueueReadImage error: %s\n", clGetErrorString(error));
            }

            mkdir("results", 0777);

            char name[256];
            snprintf(name, 256, "./results/result%03i.png", iangle);
            int res = stbi_write_png(name, RES_X, RES_Y, 4, image, RES_X * sizeof(rgba_8i));
            printf("Wrote %s\n", name);
            assert(res != 0);

            clReleaseMemObject(result_images[iangle]);
        }
        

        error = clFlush(command_queue);
        if (error != 0) {
            printf("clFlush error: %s\n", clGetErrorString(error));
        }
        error = clFinish(command_queue);
        if (error != 0) {
            printf("clFinish error: %s\n", clGetErrorString(error));
        }

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(volume_image);
        clReleaseCommandQueue(command_queue);
    }

    clReleaseContext(context);

    assert(H5close() >= 0);
    return ret_val;
}

