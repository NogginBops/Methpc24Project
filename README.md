# Methpc24Project

Project in DD2356 methpc

This project is based on Philip Mocz's volume render project in python: https://github.com/pmocz/volumerender-python.

The project consists of four directories.

```
./
 |- c_impl/
 |- mpi_impl/
 |- omp_impl/
 |- data/
```

`c_impl/` contains a straight forward single threaded c implementation of the volume renderer. See [`c_impl/README.md`](/c_impl/README.md) for more info.

`omp_impl/` contains an OpenMP implementation (very similar to the original c impl).  See [`omp_impl/README.md`](/omp_impl/README.md) for more info.

`mpi_impl/` contains an MPI implementation (using mpich).  See [`mpi_impl/README.md`](/mpi_impl/README.md) for more info.

`data/` contains the hdf5 datacube that is rendered. Copied from Philip Mocz's original project.

## Compiling on PDC Dardel

The C, OpenMP, and MPI version are possible to compile on PDC Dardel.

To compile on PDC Dardel the make files need to patched to contain the correct link paths for the HDF5 library and MPICH library. These changes are contained in the [`pdc_dardel.patch`](./pdc_dardel.patch) file that can be applied to compile on dardel.