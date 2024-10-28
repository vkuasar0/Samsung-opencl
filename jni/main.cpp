#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <functional>
#include "helper.h"


void log_times(const std::string& operation_name, std::function<double()> func, int iterations) {
    std::string filename = "report_" + operation_name + ".csv";
    std::ofstream of(filename, std::ios::app);
    double totalTime = 0.0;
    for (int i = 0; i < iterations; ++i) {
        double time = func();
        totalTime += time;
        of << time << "\n";
    }
    of << "AVERAGE TIME: " << totalTime / iterations << " microseconds\n";
    of.close();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Please provide an argument" << std::endl;
        return 1;
    }

    char *ch = argv[1];
    std::cout << "Arg: " << ch << std::endl;
    const int iterations = 100;

    if (!strcmp(ch, "0")) { // ADD
        log_times("add", add_bin, iterations);
    } else if (!strcmp(ch, "1")) { // GAUSSIAN BLUR
        log_times("gaussian_blur", gaussian_blur, iterations);
    } else if (!strcmp(ch, "2")) { // CROP
        log_times("crop", crop, iterations);
    } else if (!strcmp(ch, "3")) { // LANCZOS
        log_times("lanczos", lanczos, iterations);
    } else if (!strcmp(ch, "4")) { // EMBOSS
        log_times("emboss", emboss, iterations);
    } else if (!strcmp(ch, "5")) { // MUL
        log_times("multiply_images", multiply_images, iterations);
    } else if (!strcmp(ch, "6")) { // GRAY
        log_times("gray", gray_bgr, iterations);
    } else if (!strcmp(ch, "7")) { // CVT- RGB TO BGR SWAP
        log_times("cvt", cvt, iterations);
    } else if (!strcmp(ch, "9")) { // RESHAPE
        log_times("reshape", reshape_image, iterations);
    } else if (!strcmp(ch, "10")) { // RGB to YcbCr
        log_times("ycrcb", rgbToYCbCr, iterations);
    } else if (!strcmp(ch, "11")) {   //MEDIAN
        log_times("median",median , iterations);
    } else if (!strcmp(ch, "12")) {  //SOBEL-EDGE
        log_times("sobelEdge", sobelEdge, iterations); 
    } else if(!strcmp(ch, "13")) {  //DOWNSIZE
        log_times("downsize", downsize_image, iterations);
    } else if(!strcmp(ch, "14")) {  //ALPHA-BLENDING
        log_times("alpha_blending", perform_alpha_blending, iterations);
    } else if(!strcmp(ch, "15")) {
        log_times("solarize", solarize_image, iterations);
    } else if(!strcmp(ch, "16")) {
        log_times("pixellate", pixellate_image, iterations);
    } else if(!strcmp(ch, "17")) {
        log_times("noop", noop_image, iterations);
    } else if(!strcmp(ch, "18")) {
        log_times("nearest_neighbour", nearestNeighborImage, iterations);
    } else if(!strcmp(ch, "19")) {
        log_times("ycrcb_bgr", ycbcrToRgb, iterations);
    } else {
        std::cerr << "Invalid argument" << std::endl;
        return 1;
    }

    std::cout << "Execution times logged" << std::endl;
    return 0;
}
