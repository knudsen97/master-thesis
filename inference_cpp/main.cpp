#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include "inc/inference.h"
#include "inc/test.h"
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

/**
 * @brief Convert a torch tensor to an opencv mat
 * @param tensor The tensor to convert
 * @param opencv_mat The mat to save the converted tensor to
 * @return true if the conversion was successful, false otherwise
 */
bool convert_tensor_to_mat(const torch::Tensor& tensor, cv::Mat& opencv_mat) {
    // pytorch_mat = cv::Mat(tensor.size(0), tensor.size(1), CV_32FC3, tensor.data_ptr());
    opencv_mat = cv::Mat(tensor.size(0), tensor.size(1), CV_32FC3);
    cv::Mat opencv_mat_R(tensor.size(0), tensor.size(1), CV_32FC1);
    cv::Mat opencv_mat_G(tensor.size(0), tensor.size(1), CV_32FC1);
    cv::Mat opencv_mat_B(tensor.size(0), tensor.size(1), CV_32FC1);

    /*  opencv expect a sequence of alternating RGB but gets all R values first, then G, the B,
        so we need to read them individually, then merge them together*/
    int channels = tensor.size(2);
    int values_per_channel = tensor.size(0)*tensor.size(1);
    for (int channel = 0; channel < channels; channel++)
    {    
        for (int index = 0; index < values_per_channel; index++)
        {
            int tensor_index = index + channel * values_per_channel;
            float data = tensor.data_ptr<float>()[tensor_index];
            int row = index % tensor.size(1);
            int col = index / tensor.size(1);
            if (channel == 0)
                opencv_mat_B.at<float>(col, row) = data;
            else if (channel == 1)
                opencv_mat_G.at<float>(col, row) = data;
            else if (channel == 2)
                opencv_mat_R.at<float>(col, row) = data;
        }
    }
    cv::merge(std::vector<cv::Mat>{opencv_mat_B, opencv_mat_G, opencv_mat_R}, opencv_mat);
    return true;
}

int main() {
    cv::Mat image = cv::imread("../../data/color-input/000028-0.png");
    if (image.empty()) {
        std::cerr << "Could not read image" << std::endl;
        return -1;
    }

    std::string path_to_model = "../models/temp_model.pt";
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(path_to_model);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model at: " << path_to_model << std::endl;
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 128, 160}));

    torch::Tensor output = module.forward(inputs).toTensor();

    output = output.squeeze();
    output = output.permute({ 1, 2, 0 });
    
    cv::Mat returned_image;
    bool got;
    got = convert_tensor_to_mat(output, returned_image);
    if(!got)
    {
        std::cout << "Could not convert tensor to mat" << std::endl;
        return -1;
    }
    cv::imshow("image", returned_image);
    cv::waitKey(0);

    return 0;
}