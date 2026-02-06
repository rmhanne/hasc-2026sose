# Beware!

I am currently restructuring the code for the lecture in summer semester 2026! This read me is not up to date and the code is not ready yet.

# hasc-code

This repository collects code for the hardware aware scientific computing lecture. The code in this repository will change as the lecture proceeds so expect some changes and merge conflicts if you change anything in here.

# Software requirements

The hasc-code examples are tested on Linux (Ubuntu 24.04.3 (Noble Numbat) as well as on Apple MacBook Pro (M2 Max, MacOS Ventura, MacPorts) using the GNU C++ Compiler. Windows is not supported.

The lecture covers several different programming models. Some of these are handled by most C++ compilers, others require additional software to be installed on your system. Some of this software can be installed via usual packet managers, other has to be downloaded and installed manually. The following subsection should give the rquired information.

## Simd vectorization for Intel/AMD processors 

Explicit SIMD vectorization for AVX2 and AVX512 is done using the [vector class library](https://github.com/vectorclass/version2). Use the --recursive option when cloning the repository.

## Simd vectorization for Arm (neon) processors 

Explicit SIMD vectorization for Arm processors, more specifically Apple Silicon with NEON support is done using neon intrinsics [(see here for documentation)](https://github.com/thenifty/neon-guide). This should be available with gcc on such systems without installation.

## Portable Simd vectorization using std::simd 

This requires a C++-compiler supporting the 2026 standard of C++. As of this writing this is supported by GCC 14 when passing the option -std=c++26 in the namespace std::experimental.

## Open MP 

Is supported by GCC and Clang compilers using appropriate options (see make.def)

## Intel Thread Building Blocks (TBB)

This is an open-source library released by Intel. Hasc-code uses the newer version named oneTBB. It should be available with your package manager. With macports the port is called onetbb.

On LINUX you should install Intel's [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline), which gives you oneTBB as well as the Sycl compiler (see below).

## Sycl

[SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) is C++ extension based on an open standard that allows one to write portable code for CPUs and GPUs (at least that is the idea). [Several compilers]((https://www.intel.com/content/www/us/en/developer/articles/technical/quick-guide-to-sycl-implementations.html)) are available. Hasc-code uses Intel's [oneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline) for its SYCL examples. This is restricted to Intel/AMD hardware.

## Message Passing Interface (MPI)

Should be available with most package managers. Popular implementations are [MPICH](https://www.mpich.org/) and [OpenMPI](https://www.open-mpi.org/) (not to be confused with OpenMP).




# Installation

Download the code from the git repository using

```
git clone --recursive https://parcomp-git.iwr.uni-heidelberg.de/Teaching/hasc-code.git
```
The `--recursive` option is important to include the [vector class library](https://github.com/vectorclass/version2) as submodule.


Have a look at the `Makefile` and adjust it to your system:

- Choose a compiler
- Adjust the compiler flags
- Adjust the linker flags

If everything is set up correctly you should be able to build the examples by typing

```
make
```

## CMake

As an alternative you can use CMake to create an out of source build. If you want to change the compiler flags you can do so in the file `CMakeLists.txt`. The steps to build the executables are as follows:

```
mkdir build
cd build
cmake ..
make
```

Usually it is enough to just call `make` in the build directory to rebuild executables. If it doesn't behave as it should you could try removing the build directory and creating a new one.

For some executables you need to pass special options to cmake:

If you want to build executables that use Intel MKL you have to make sure that Intel MKL is installed and added to your paths (usually done by calling a setup script from intel). Then you can call CMake from the build directory with

```
cmake -DHASC_HAVE_MKL=ON ..
```

For executables using SYCL you need a compiler that supports sycl and call cmake like this

```
cmake -DHASC_HAVE_SYCL=ON ..
```

For using a specific compiler (in case you have multiple on your system) you can run cmake likes this

```
cmake -DHASC_HAVE_MKL=ON -DHASC_HAVE_SYCL=ON -DCMAKE_CXX_COMPILER=dpcpp ..
cmake -DHASC_HAVE_MKL=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
```


## Troubleshooting

In case of problems you can try to build the executables with

```
make VERBOSE=1
```

This way you will see all commands that get executed. This way you can check if the compiler and the compile and link flags look reasonable.
