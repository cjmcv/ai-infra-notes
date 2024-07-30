adb push ./bin/pai_cl /data/local/tmp/pai/bin/pai_cl
adb push ./kernels /data/local/tmp/pai/kernels

adb shell "chmod 777 -R /data/local/tmp/pai/ && cd /data/local/tmp/pai/ && export LD_LIBRARY_PATH=/system/lib64/:/system/vender/lib64/:$LD_LIBRARY_PATH && ./bin/pai_cl"