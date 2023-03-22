#ifndef INFERENCE_H
#define INFERENCE_H

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

// #include <torch/torch.h>
// #include <torch/script.h> // One-stop header.




class Inference
{
public:

    /**
     * @brief Construct a new Inference object. The model path is set to the default path "../../models/first_model.pt".
    */
    Inference();

    /**
     * @brief Construct a new Inference object.
     * @param model_path The path to the model to load.
    */
    Inference(std::string model_path);
    ~Inference();

    /**
     * @brief Predict the output of an input image.
     * @param input The input image.
     * @param output The output image.
     * @return True if the prediction was successful, false otherwise.
    */
    bool predict(const cv::Mat input, cv::Mat& output);

    /**
     * @brief Normalize an image.
     * @param image The image to normalize.
     * @param mean The mean to subtract from the image.
     * @param std The standard deviation to divide the image by.
     * @return True if the normalization was successful, false otherwise.
    */
    static bool normalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std);

    /**
     * @brief Denormalize an image.
     * @param image The image to denormalize.
     * @param mean The mean to add to the image.
     * @param std The standard deviation to multiply the image by.
     * @return True if the denormalization was successful, false otherwise.
    */
    static bool denormalize_image(cv::Mat& image, std::vector<float> mean, std::vector<float> std);

private:
    std::string model_path;
    // torch::jit::script::Module model;

};

#endif // INFERENCE_H

