LOCAL_PATH := $(call my-dir)

CVROOT := ../include/sdk/native/jni


include $(CLEAR_VARS)
LOCAL_MODULE := opencl
LOCAL_SRC_FILES := ../include/libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=STATIC
include $(CVROOT)/OpenCV.mk

LOCAL_MODULE += myModule

LOCAL_C_INCLUDES += ../include/
LOCAL_SRC_FILES += main.cpp helper.cpp
LOCAL_SHARED_LIBRARIES := opencl
LOCAL_CFLAGS += -std=c++11 -frtti -fexceptions  -w
# LOCAL_LDLIBS += -llog -L$(SYSROOT)/usr/lib
LOCAL_LDLIBS += -llog

include $(BUILD_EXECUTABLE)
