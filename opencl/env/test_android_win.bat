adb shell "rm -r /data/local/tmp/pai/"

adb push ./bin/pai_cl /data/local/tmp/pai/bin/pai_cl
adb push ./kernels /data/local/tmp/pai/kernels

adb shell "chmod 777 -R /data/local/tmp/pai/ && cd /data/local/tmp/pai/ && export LD_LIBRARY_PATH=/system/lib/:/system/vender/lib/:$LD_LIBRARY_PATH && ./bin/pai_cl"