
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include "inc/inference.h"
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <algorithm>

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
    opencv_mat.convertTo(opencv_mat, CV_8UC3, 255);
    // std::cout << "opencv_mat: \n" << opencv_mat << std::endl;
    return true;
}

/**
 * @brief Convert an opencv mat to a torch tensor. This also divides the values by 255.
 * @param mat The mat to convert.
 * @param tensor The tensor to save the converted mat to.
 * @return true if the conversion was successful, false otherwise.
*/
bool mat_to_tensor(const cv::Mat& img, torch::Tensor& tensor)
{
    // Convert the cv::Mat to a pytorch tensor
    tensor = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kFloat);
    tensor = tensor.permute({ 2, 0, 1 });
    tensor = tensor.unsqueeze(0);
    return true;
}

/**
 * @brief Run inference on the model.
 * @param input The input image.
 * @param returned_image The image returned by the model.
 * @param path_to_model The path to the model. Default is "../models/temp_model.pt".
 * @return true if the inference was successful, false otherwise.
*/
bool inference(const cv::Mat& input, cv::Mat& returned_image, std::string path_to_model)
{
    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(path_to_model);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model at: " << path_to_model << std::endl;
        return false;
    }

    // Convert the input image to a tensor
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor tensor;
    mat_to_tensor(input, tensor);
    inputs.push_back(tensor);
    // std::cout << "input shape: " << input.size() << std::endl;
    // std::cout << "tensor shape: " << tensor.sizes() << std::endl;

    // Run the model and turn its output into a tensor
    torch::Tensor output = module.forward(inputs).toTensor();

    // Convert the tensor to a cv::Mat
    output = output.squeeze();
    output = output.permute({ 1, 2, 0 });
    convert_tensor_to_mat(output, returned_image);
    
    return true;
}

/**
 * @brief Run inference on the model.
 * @param input The input image.
 * @param returned_image The image returned by the model.
 * @param path_to_model The path to the model. Default is "../models/temp_model.pt".
 * @return true if the inference was successful, false otherwise.
*/
bool inference(const torch::Tensor& input, torch::Tensor& returned_image, std::string path_to_model)
{
    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(path_to_model);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model at: " << path_to_model << std::endl;
        return false;
    }

    // Convert the input image to a tensor
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor tensor;
    inputs.push_back(tensor);
    // std::cout << "input shape: " << input.size() << std::endl;
    // std::cout << "tensor shape: " << tensor.sizes() << std::endl;

    // Run the model and turn its output into a tensor
    torch::Tensor output = module.forward(inputs).toTensor();

    // Convert the tensor to a cv::Mat
    output = output.squeeze();
    output = output.permute({ 1, 2, 0 });
    returned_image = output;
    
    return true;
}

/**
 * @brief Normalize an image.
 * @param image The image to normalize.
 * @param mean The mean to subtract from the image.
 * @param std The standard deviation to divide the image by.
 * @return true if the normalization was successful, false otherwise.
*/
bool normalize_image(cv::Mat& image, const std::vector<float>& mean, const std::vector<float>& std)
{
    if (image.empty())
    {
        std::cerr << "Error: image is empty." << std::endl;
        return false;
    }
    if (mean.size() != 3 || std.size() != 3)
    {
        std::cerr << "Error: mean and std must be of size 3." << std::endl;
        return false;
    }
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    image = image - cv::Scalar(mean[0], mean[1], mean[2]);
    image = image / cv::Scalar(std[0], std[1], std[2]);
    return true;
}
/**
 * @brief Normalize an image.
 * @param image The image to normalize.
 * @param mean The mean to subtract from the image.
 * @param std The standard deviation to divide the image by.
 * @return true if the normalization was successful, false otherwise.
*/
bool normalize_image(cv::Mat& image, const std::array<float, 3>& mean, const std::array<float, 3>& std)
{
    if (image.empty())
    {
        std::cerr << "Error: image is empty." << std::endl;
        return false;
    }
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    image = image - cv::Scalar(mean[0], mean[1], mean[2]);
    image = image / cv::Scalar(std[0], std[1], std[2]);
    return true;
}

/**
 * @brief Denormalize an image.
 * @param image The image to denormalize.
 * @param mean The mean to add to the image.
 * @param std The standard deviation to multiply the image by.
 * @return true if the denormalization was successful, false otherwise.
*/
bool denormalize_image(cv::Mat& image, const std::vector<float> mean, const std::vector<float> std)
{
    if (image.empty())
    {
        std::cerr << "Error: image is empty." << std::endl;
        return false;
    }
    if (mean.size() != 3 || std.size() != 3)
    {
        std::cerr << "Error: mean and std must be of size 3." << std::endl;
        return false;
    }
    cv::Mat bands[3], merged;
    split(image, bands);
    for (int i = 0; i < 3; i++)
    {
        bands[i] = bands[i] * std[i];
        bands[i] = bands[i] + mean[i];
    }
    cv::merge(bands,3, merged);
    return true;
}

/**
 * @brief Denormalize an image.
 * @param image The image to denormalize.
 * @param mean The mean to add to the image.
 * @param std The standard deviation to multiply the image by.
 * @return true if the denormalization was successful, false otherwise.
*/
bool denormalize_image(cv::Mat& image, const std::array<float, 3> mean, const std::array<float, 3> std)
{
    if (image.empty())
    {
        std::cerr << "Error: image is empty." << std::endl;
        return false;
    }
    cv::Mat bands[3], merged;
    split(image, bands);
    for (int i = 0; i < 3; i++)
    {
        bands[i] = bands[i] * std[i];
        bands[i] = bands[i] + mean[i];
    }
    cv::merge(bands,3, merged);
    return true;
}

int main(int argc, char** argv){
    std::string path_to_model;
    if (argc == 2)
    {
        path_to_model = argv[1];
    }
    else
    {
        path_to_model = "../../../../models/unet_resnet101_1_jit.pt";
    }
    // define mean and std_dev
    std::array<float, 3> mean = {0.2966, 0.3444, 0.4543}; // mean = [0.4543, 0.3444, 0.2966] [R, G, B]
    std::array<float, 3> std_dev = {0.2198, 0.2415, 0.2423};// std_dev = [0.2198, 0.2415, 0.2423] [R, G, B]

    // read image
    cv::Mat input_image = cv::imread("input.png", cv::IMREAD_COLOR);
    // cv::imshow("input", input_image);
    // cv::waitKey(0);
    if (input_image.empty())
    {
        std::cerr << "Error: could not read input.png." << std::endl;
        return -1;
    }
    cv::Vec <uchar, 3> debug;
    // transform image
    cv::Mat input_image_resized;
    torch::Tensor input_tensor;

    cv::resize(input_image, input_image, cv::Size(160, 128));
    normalize_image(input_image, mean, std_dev);
    // mat_to_tensor(input_image, input_tensor);
    // cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    // inference
    cv::Mat returned_image;
    if (!inference(input_image, returned_image, path_to_model))
    {
        std::cerr << "Error: inference failed." << std::endl;
        return -1;
    }

    // reverse transform image
    cv::cvtColor(returned_image, returned_image, cv::COLOR_RGB2BGR);
    cv::resize(returned_image, returned_image, cv::Size(640, 480));
    denormalize_image(returned_image, mean, std_dev);

    // save image
    cv::imwrite("output.png", returned_image);
    return 0;
}