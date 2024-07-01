#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

#ifndef AOCLIP__H
#define AOCLIP__H

std::string loadKernel(const char *filename);
double add_bin();
double gaussian_blur();
double crop();
double lanczos();
double emboss();
double multiply_images();
double gray_bgr();
double cvt();
double reshape_image();

#endif //SAMPLE_OPENCL_NDK_MASTER_COPY_BLUR_GPU_H
