#include <iostream>
#include <opencv2/opencv.hpp>
#include "inc/inference.h"
#include "inc/test.h"
#include <chrono>


int main() {
    cv::Mat image = cv::imread("../../data/color-input/000028-0.png");
    if (image.empty()) {
        std::cerr << "Could not read image" << std::endl;
        return -1;
    }

    Inference inf("../py_scripts/LoadModel.py", "Inference", "LoadModel");

    cv::Mat returned_image;
    cv::transpose;
    auto t_start = std::chrono::high_resolution_clock::now();
    bool sucess = inf.predict<3, uint8_t>(image, returned_image);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms" << std::endl;
    if (!sucess) {
        std::cerr << "Error: could not predict" << std::endl;
        return -1;
    }
    // std::cout << "transformation:" << std::endl;
    // std::cout << transformation << std::endl;
    cv::imshow("cpp inf image", returned_image);
    cv::waitKey(0);

    return 0;
}