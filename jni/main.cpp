#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <string.h>

#include "helper.h"

void testing(){
    init();
    auto start1 = std::chrono::high_resolution_clock::now();
    add_bin();
    auto stop1 = std::chrono::high_resolution_clock::now();
    //std::cout << "CV GPU ADD TIME IS"<<(stop1-start1);
    auto int_us1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::chrono::duration<long, std::micro> int_usec1 = int_us1;
    std::cout << "Total" << int_usec1.count() / 1000.0 <<std::endl;
    exit(0);
}

int main(int argc, char *argv[]) {

    char *ch;
    ch = argv[1];
    printf("Arg: %s\n", ch);

    if (!strcmp(ch, "0")) { // ADD
        std::ofstream of;
        const char *write_filename = "report.csv";

        // opencl
        auto start = std::chrono::high_resolution_clock::now();
        add_bin();
        auto stop = std::chrono::high_resolution_clock::now();
        auto int_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        of.open(write_filename, std::ios::app);
        of << "," << int_us.count() / 1000.0 << "\n";
        of.close();
    }

    else if (!strcmp(ch, "1")) { // GAUSSIAN BLUR
        std::ofstream of;
        const char *write_filename = "report.csv";

        // opencl
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_blur();
        auto stop = std::chrono::high_resolution_clock::now();
        auto int_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        of.open(write_filename, std::ios::app);
        of << "," << int_us.count() / 1000.0 << "\n";
        of.close();
    }
}