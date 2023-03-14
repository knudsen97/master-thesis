#include <iostream>
#include <opencv2/opencv.hpp>
#include "inc/inference.h"
#include "inc/test.h"



int main() {
    cv::Mat image = cv::imread("../../data/color-input/000000-0.png");
    if (image.empty()) {
        std::cerr << "Could not read image" << std::endl;
        return -1;
    }

    Inference inf("../py_scripts/PredictionProcessor.py", "calculate_transformation", "load_model");

    cv::Mat returned_image;
    bool sucess = inf.predict<3, uint8_t>(image, returned_image);
    if (!sucess) {
        std::cerr << "Error: could not predict" << std::endl;
        return -1;
    }
    // std::cout << "transformation:" << std::endl;
    // std::cout << transformation << std::endl;
    cv::imshow("transformation", returned_image);
    cv::waitKey(0);

    std::cout << "test:" << std::endl;
    test t;
    int r = t.add<int>(1, 2);
    std::cout << r << std::endl;



    return 0;
}