
:: x86 win10 opencl

set OPENCL_SDK_INCLUDE=D:/software/oneAPI/compiler/2024.0/include/sycl
set OPENCL_SDK_LIB=D:/software/oneAPI/compiler/2024.0/lib

mkdir build-win-x86
pushd build-win-x86
cmake -G "MinGW Makefiles" -DOPENCL_SDK_INCLUDE:STRING=%OPENCL_SDK_INCLUDE% -DOPENCL_SDK_LIB:STRING=%OPENCL_SDK_LIB% ..
mingw32-make -j8
popd