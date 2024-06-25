#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#ifndef AOCLIP__H
#define AOCLIP__H

//testing
void init();
std::string read_file(const char *filename);
std::string loadKernel(const char *filename);
void add_bin();
int gaussian_blur();
int multiply_images();

#endif //SAMPLE_OPENCL_NDK_MASTER_COPY_BLUR_GPU_H
