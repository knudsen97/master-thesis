#include <iostream>
#include <opencv2/opencv.hpp>
#include "inc/inference.h"
#include "inc/test.h"

int main() {
    cv::Mat image = cv::imread("../_data/color-input/000000-0.png");

    Inference inf("py_scripts/inference.py");

    cv::Mat2f prediction;
    prediction = inf.predict<1>(image);

    // test t;
    // int r = t.add<int>(1, 2);
    // std::cout << r << std::endl;



    return 0;
}