#include <iostream>
#include <stdlib.h>
#include <random>
#include <string.h>

#include "helper.h"

int main(int argc, char *argv[]) {

    char *ch;
    ch = argv[1];
    printf("Arg: %s\n", ch);
    std::ofstream of;
    const char *write_filename = "report.csv";

    if (!strcmp(ch, "0")) { // ADD
        double time = add_bin();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "1")) { // GAUSSIAN BLUR
        double time = gaussian_blur();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "2")) { // CROP
        double time = crop();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "3")) { // lanczos
        double time = lanczos();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "4")) { // emboss
        double time = emboss();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "5")) { // MUL
        double time = multiply_images();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "6")) { // GRAY
        double time = gray_bgr();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "7")) { // CVT- RGB TO BGR SWAP
        double time = cvt();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
     else if (!strcmp(ch, "9")) { // RESHAPE
        double time = reshape_image();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "14")) { // DOWNSIZE NNI
        double time = downsize_image();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "16")) { // DOWNSIZE BICUBIC
        double time = downsize_bicubic();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "11")) { // RGB TO YCbCr
        double time = rgbToYCbCr();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "12")) { // YCbCr to RGB
        double time = ycbcrToRgb();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }
    else if (!strcmp(ch, "19")) { // SOLARIZE
        double time = solarize_image();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "20")) { // PIXELLATE
        double time = pixellate_image();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "21")) { // NEAREST NEIGHBOR
        double time = nearestNeighborImage();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "22")) { // NOOP
        double time = processImage();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "23")) { // TEXTURE
        double time = texture();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "25")) { // SOBEL EDGE
        double time = sobelEdge();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    else if (!strcmp(ch, "27")) { // MEDIAN FILTER
        double time = median_filter_image();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    of.close();
}
