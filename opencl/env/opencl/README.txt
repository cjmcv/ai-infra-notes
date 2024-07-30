# 以32位为例

1. 执行脚本pull_cl_lib32_win.bat，将手机内相关库拷贝出来到文件加lib中；

2. build.sh中开启android armv7部分代码。

3. 检查build.sh中的 ANDROID_NDK / OPENCL_HEADER 和 OPENCL_LIBS 是否正确。

   OPENCL_HEADER 下载自 https://github.com/KhronosGroup/OpenCL-Headers。
   
   OPENCL_LIBS 对应 pull_cl_lib32_win.bat 拷贝存放路径。

4. ubuntu 端执行./build.sh 生成可执行文件。

5. 将可执行文件拷贝到 /data/local/tmp/ 中执行，需要注意添加库搜索路径 /system/lib/:/system/vender/lib/