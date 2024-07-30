:: 比32位多依赖 ld-android.so
:: ? libc.so: undefined reference to `android_get_application_target_sdk_version@LIBC_N

md lib64

adb pull /system/vendor/lib64/libOpenCL.so ./lib64
adb pull /system/lib64/libutils.so ./lib64
adb pull /system/lib64/libcutils.so ./lib64
adb pull /system/lib64/liblog.so ./lib64
adb pull /system/lib64/libprocessgroup.so ./lib64
adb pull /system/lib64/libvndksupport.so ./lib64
adb pull /system/lib64/libbase.so ./lib64
adb pull /system/lib64/libcgrouprc.so ./lib64
adb pull /system/lib64/libdl_android.so ./lib64
adb pull /system/lib64/libc++.so ./lib64
adb pull /system/lib64/libc.so ./lib64
adb pull /system/lib64/ld-android.so ./lib64

pause