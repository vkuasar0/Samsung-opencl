#!/bin/bash
adb push ../libs/arm64-v8a/. /data/local/tmp
adb push ../assets/cvtColor_bgr.cl /data/local/tmp
adb push ../assets/gray_bgr.cl /data/local/tmp
adb push ../assets/ycrcb.cl /data/local/tmp
adb push ../assets/ycrcb_bgr.cl /data/local/tmp

adb push ../assets/gaussianBlur.cl /data/local/tmp
adb push ../assets/boxTrial.cl /data/local/tmp

adb push ../assets/count_nonzero.cl /data/local/tmp

adb push ../assets/reshape.cl /data/local/tmp
adb push ../assets/crop.cl /data/local/tmp
adb push ../assets/pyrup.cl /data/local/tmp
adb push ../assets/pyrdown.cl /data/local/tmp
adb push ../assets/normalize.cl /data/local/tmp

adb push ../assets/add.cl /data/local/tmp
adb push ../assets/absdiff.cl /data/local/tmp
adb push ../assets/sub.cl /data/local/tmp
adb push ../assets/mul.cl /data/local/tmp
adb push ../assets/div.cl /data/local/tmp
adb push ../assets/mean.cl /data/local/tmp
adb push ../assets/split.cl /data/local/tmp
adb push ../assets/merge.cl /data/local/tmp
adb push ../assets/equal.cl /data/local/tmp
adb push ../assets/upsize_bicubic.cl /data/local/tmp
adb push ../assets/upsize_nni.cl /data/local/tmp
adb push ../assets/boxblur.cl /data/local/tmp
adb push ../assets/downsize_bicubic.cl /data/local/tmp
adb push ../assets/downsize_nni.cl /data/local/tmp
adb push ../assets/add.cl /data/local/tmp
adb push ../assets/add_image.cl /data/local/tmp
adb push ../image_add1.png /data/local/tmp
adb push ../image_add2.png /data/local/tmp
adb push ../gray_input.png /data/local/tmp
adb push ../ycrcb_input.png /data/local/tmp
adb push ../output1.png /data/local/tmp
adb push ../output2.png /data/local/tmp
adb push ../output3.png /data/local/tmp
adb push ../instagram.png /data/local/tmp
adb push ../output/report.csv /data/local/tmp
adb push ../shell/runapis.sh /data/local/tmp

adb shell
