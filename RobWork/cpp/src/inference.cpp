#include "../inc/inference.hpp"

Inference::Inference()
{
    this->model_path = "../../models/first_model.pt";
    // this->model = torch::jit::load(model_path);
}

Inference::Inference(std::string model_path)
{
    this->model_path = model_path;
    // this->model = torch::jit::load(model_path);
}

Inference::~Inference()
{
}

bool Inference::predict(const cv::Mat input, cv::Mat& output)
{
    // cv::Mat input_image(480, 640, CV_8UC3, CV_RGB(255,255,255));
    cv::imwrite("input.png", input);
    std::string command;
    command = "./inference " + this->model_path;
    int a = system(command.c_str());
    if (a != 0)
    {
        std::cout << "Error: system call failed." << std::endl;
        return false;
    }
    cv::Mat returned_image = cv::imread("output.png");
    output = returned_image;
    return true;
}


bool Inference::normalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std)
{
    if (image.empty())
    {
        std::cout << "Error: image is empty." << std::endl;
        return false;
    }
    if (mean.size() != 3 || std.size() != 3)
    {
        std::cout << "Error: mean and std must be of size 3." << std::endl;
        return false;
    }
    image = image - cv::Scalar(mean[0], mean[1], mean[2]);
    image = image / cv::Scalar(std[0], std[1], std[2]);
    return true;
}

bool Inference::denormalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std)
{
    if (image.empty())
    {
        std::cout << "Error: image is empty." << std::endl;
        return false;
    }
    if (mean.size() != 3 || std.size() != 3)
    {
        std::cout << "Error: mean and std must be of size 3." << std::endl;
        return false;
    }
    image = image * cv::Scalar(std[0], std[1], std[2]);
    image = image + cv::Scalar(mean[0], mean[1], mean[2]);
    return true;
}