#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// #include "inc/inference.h"
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <algorithm>
#include "inc/test.h"

/**
 * @brief Convert a torch tensor to an opencv mat.
 * @param tensor The tensor to convert.
 * @param opencv_mat The mat to save the converted tensor to.
 * @return true if the conversion was successful, false otherwise.
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
bool mat_to_tensor(const cv::Mat img, torch::Tensor& tensor)
{
    // Convert the cv::Mat to a pytorch tensor
    tensor = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
    tensor = tensor.permute({ 2, 0, 1 });
    tensor = tensor.toType(torch::kFloat);
    tensor = tensor.div(255);
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
bool inference(cv::Mat input, cv::Mat& returned_image, std::string path_to_model = "../models/temp_model.pt")
{
    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(path_to_model);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model at: " << path_to_model << std::endl;
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
 * @brief Normalize an image.
 * @param image The image to normalize.
 * @param mean The mean to subtract from the image.
 * @param std The standard deviation to divide the image by.
 * @return true if the normalization was successful, false otherwise.
*/
bool normalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std)
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

/**
 * @brief Denormalize an image.
 * @param image The image to denormalize.
 * @param mean The mean to add to the image.
 * @param std The standard deviation to multiply the image by.
 * @return true if the denormalization was successful, false otherwise.
*/
bool denormalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std)
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

/** 
 * @brief Run a command in the terminal.
 * 
 * Kudos to waqas for this function: https://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix
 * 
 * @param cmd The command to run.
 * @return The output of the command.
*/
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::stringstream bin_file_test(std::stringstream& ss)
{
    std::string str;
    // we expect the image to be 480x640 BGR
    std::array<u_char, 480*640*3> data;
    int index = 0;
    while( ss >> str){
        if (str != "[" && str != "]")
        {
            if(str[str.size()-1] == ',')
            {
                str.pop_back();
            }
            data[index] = std::stoi(str);
            index++;
        }
    }
    cv::Mat image(480, 640, CV_8UC3, data.data());
    cv::resize(image, image, cv::Size(128, 160));
    cv::Mat returned_image;
    inference(image, returned_image);
    cv::resize(returned_image, returned_image, cv::Size(640, 480));
    std::stringstream rss;
    rss << returned_image;
    // remove first and last bracket
    std::string s = rss.str();
    s = s.substr(1, s.size() - 2);
    rss.str(s);
    return rss;
}


void bin_file()
{
    std::string str;
    // we expect the image to be 480x640 BGR
    std::array<u_char, 480*640*3> data = {0};
    int index = 0;
    while( std::cin >> str){
        if (str != "[" && str != "]")
        {
            if(str[str.size()-1] == ',')
            {
                str.pop_back();
            }
            data[index] = std::stoi(str);
            index++;
        }
    }
    cv::Mat image(480, 640, CV_8UC3, data.data());
    cv::resize(image, image, cv::Size(128, 160));
    cv::Mat returned_image;
    inference(image, returned_image);
    cv::resize(returned_image, returned_image, cv::Size(640, 480));
    std::stringstream rss;
    rss << returned_image;
    // remove first and last bracket
    std::string s = rss.str();
    s = s.substr(1, s.size() - 2);
    // rss.str(s);
    std::cout << s << std::endl;
}


bool test_inference()
{
    cv::Mat image(480, 640, CV_8UC3, CV_RGB(1,1,1));
    std::stringstream ss;
    ss << image;
    std::string s = ss.str();
    //remove first and last bracket
    s = s.substr(1, s.size() - 2);

    // send command to bin file
    std::string command = "echo " + s + " | " + "./bin_file ";
    // std::cout << s << std::endl;
    // command = "ls | grep cpp";
    std::string result = exec(command.c_str());

    // std::stringstream rss = bin_file_test(ss);
    std::stringstream rss;
    rss.str(result);
    std::string str;
    // we expect the image to be 480x640 BGR
    std::array<float, 480*640*3> data;
    int index = 0;
    while( rss >> str) {
        if (str != "[" && str != "]") {
            if(str[str.size()-1] == ',') {
                str.pop_back();
            }
            data[index] = std::stoi(str);
            index++;
        }
    }

    // convert string to cv::Mat
    cv::Mat returned_image(480, 640, CV_32FC3, data.data());

    cv::imshow("image", returned_image);
    cv::waitKey(0);


}

/**
 * @brief Test cin >> cout functionality.
*/
void bin_test()
{
    std::string str;
    // we expect the image to be 480x640 BGR
    std::array<u_char, 10> data = {0};
    int index = 0;
    while( std::cin >> str){
        if (str != "[" && str != "]")
        {
            if(str[str.size()-1] == ',')
            {
                str.pop_back();
            }
            data[index] = std::stoi(str);
            index++;
        }
    }
    for (int i = 0; i < 10; i++)
    {
        std::cout << static_cast<unsigned>(data[i]) << std::endl;
    }
}

/**
 * @brief Test the bin file generated from bin_test.
*/
void simple_test()
{
    // 480, 640
    cv::Mat image(100, 100, CV_8UC3, CV_RGB(1,1,1));
    cv::Mat* image_ptr = &image;
    std::stringstream ss;
    ss << image_ptr;
    std::string s = ss.str();

    // send command to bin file
    std::string command = "echo " + s + " | " + "./bin_file ";

    std::string result;
    // result = exec("echo 1, 2, 3, 5, 7 | ./bin_file");
    result = exec(command.c_str());

    std::stringstream rss;
    rss.str(result);
    std::string str;
    std::array<float, 480*640*3> data;
    int index = 0;
    while( rss >> str) {
        if (str != "[" && str != "]") {
            if(str[str.size()-1] == ',') {
                str.pop_back();
            }
            data[index] = std::stoi(str);
            index++;
        }
    }

    // convert string to cv::Mat
    cv::Mat returned_image(480, 640, CV_32FC3, data.data());

    cv::imshow("image", returned_image);
    cv::waitKey(0);
}

void wonky_solution_bin()
{
    // define mean and std_dev
    std::vector<float> mean = {0.2966, 0.3444, 0.4543}; // mean = [0.4543, 0.3444, 0.2966] [R, G, B]
    std::vector<float> std_dev = {0.2423, 0.2415, 0.2198};// std_dev = [0.2198, 0.2415, 0.2423] [R, G, B]

    // read image
    cv::Mat input_image = cv::imread("input.png");

    // transform image
    cv::Mat input_image_resized;
    cv::resize(input_image, input_image_resized, cv::Size(128, 160));
    normalize_image(input_image_resized, mean, std_dev);

    // inference
    cv::Mat returned_image;
    inference(input_image_resized, returned_image);

    // reverse transform image
    cv::Mat returned_image_resized;
    cv::resize(returned_image, returned_image_resized, cv::Size(640, 480));
    denormalize_image(returned_image_resized, mean, std_dev);

    // save image
    cv::imwrite("output.png", returned_image_resized);
}

void wonky_solution_test()
{
    cv::Mat input_image(480, 640, CV_8UC3, CV_RGB(255,255,255));
    cv::imwrite("input.png", input_image);
    int a = system("./wonky_solution_bin");
    std::cout << a << std::endl;
    cv::Mat returned_image = cv::imread("output.png");
    cv::imshow("image", returned_image);
    cv::waitKey(0);
}

/**
 * TODO: normalize image with mean and std from dataset.
 * TODO: denormalize image.
 * TODO: make test function to test the generated binary file.
 * TODO: remove start and end bracket from cin.
*/
int main() {
    wonky_solution_test();
    // wonky_solution_bin();
    // bin_test();
    // simple_test();
    // bin_file();
    // test_inference();
    return 0;
}