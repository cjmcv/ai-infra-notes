#!/usr/bin/env bash

############################################
## x86 no gpu
# mkdir -p build-x86
# pushd build-x86
# cmake -DCMAKE_BUILD_TYPE=DEBUG ..
# make -j8
# popd

############################################
## android opencl
ANDROID_NDK=/home/shared_dir/android-ndk-r21e/
OPENCL_HEADER=/home/shared_dir/pai-dev/example/cl/opencl/OpenCL-Headers/

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
OPENCL_LIBS=/home/shared_dir/pai-dev/example/cl/opencl/lib/
cmake -DOPENCL_SDK_INCLUDE:STRING=$OPENCL_HEADER -DOPENCL_SDK_LIB:STRING=$OPENCL_LIBS \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-21 ..
make -j8
# make install
popd

# ##### android aarch64
# mkdir -p build-android-aarch64
# pushd build-android-aarch64
# OPENCL_LIBS=/home/shared_dir/pai-dev/example/cl/opencl/lib64/
# cmake -DOPENCL_SDK_INCLUDE:STRING=$OPENCL_HEADER -DOPENCL_SDK_LIB:STRING=$OPENCL_LIBS \
#       -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#       -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-24 ..
# make -j4
# # make install
# popd