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

     else if (!strcmp(ch, "4")) { // MUL
        double time = multiply_images();
        of.open(write_filename, std::ios::app);
        of << "," << time << "\n";
    }

    of.close();
}
