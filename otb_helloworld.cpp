//
// Created by auroua on 16-6-14.
//
#include <iostream>
#include "otbImage.h"

int main(int argc, char *argv[]) {
    typedef otb::Image<unsigned short, 2> ImageType;
    ImageType::Pointer image = ImageType::New();
    std::cout << "OTB Hello World !" << std::endl;
    return EXIT_SUCCESS;
}