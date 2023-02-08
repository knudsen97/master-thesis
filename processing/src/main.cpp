// Include OpenCV 
// #include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


#include "../inc/PredictionProcessor.hpp"


void load_image(const std::string &path, cv::Mat &dest, int flag = cv::IMREAD_COLOR)
{
    dest = cv::imread(path, flag);
    if (dest.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
    }
}

void load_txt(const std::string &path, cv::Mat &dest, int rows, int cols)
{
    std::ifstream file(path);
    // Resize cv::Mat to the correct size
    dest = cv::Mat(rows, cols, CV_64F);

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            double val;
            file >> val;
            dest.at<double>(i, j) = val;
        }
    }
    file.close();
}

void convert_to_gt_mask(const cv::Mat &src, cv::Mat &dest)
{
    // Convert [0,0,0] to [255,0,0], [128,128,128] to [0,255,0] and [255,255,255] to [0,0,255]
    for (int col = 0; col < src.cols; col++)
    {
        for (int row = 0; row < src.rows; row++)
        {
            if (src.at<cv::Vec3b>(row, col) == cv::Vec3b(0, 0, 0))
            {
                dest.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
            }
            else if (src.at<cv::Vec3b>(row, col) == cv::Vec3b(128, 128, 128))
            {
                dest.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 255, 0);
            }
            else if (src.at<cv::Vec3b>(row, col) == cv::Vec3b(255, 255, 255))
            {
                dest.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 0, 0);
            }
        }
    }
}

cv::Point3d pixel2World(const cv::Point2i &center, const cv::Mat &intrinsics, const double &depth)
{
    cv::Point3d center3d;
    center3d.x = (center.x - intrinsics.at<double>(0, 2)) * depth / intrinsics.at<double>(0, 0);
    center3d.y = (center.y - intrinsics.at<double>(1, 2)) * depth / intrinsics.at<double>(1, 1);
    center3d.z = depth;
    return center3d;
}

int main()
{
    const std::string idName = "000028-0";
    // Create empty image
    cv::Mat image;
    load_image("../../data/label/" + idName + ".png", image, cv::IMREAD_COLOR);

    // Create empty image for ground truth mask (Pretending this is our model prediction)
    cv::Mat prediction = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    convert_to_gt_mask(image, prediction);   

    // Create PredictionProcessor object with custom HSV bounds
    PredictionProcessor predictionProcessor(cv::Scalar(40,30,30), cv::Scalar(80,255,255));
    
    // Find the largest green objects
    std::vector<cv::Point2i> centers;
    predictionProcessor.computeCenters(prediction, centers);
    
    // Create two circles at the center of the two largest green objects
    int circleId = 5;
    cv::circle(prediction, centers[circleId], 10, cv::Scalar(255, 0, 255), 2);
    // cv::circle(prediction, centers[1], 5, cv::Scalar(255, 0, 255), 2);

    // Load camera intrinsics and camera extrinsics
    cv::Mat cameraIntrinsics, cameraExtrinsincs;
    load_txt("../../data/camera-intrinsics/" + idName + ".txt", cameraIntrinsics, 3, 3);
    load_txt("../../data/camera-pose/" + idName + ".txt", cameraExtrinsincs, 4, 4);


    // Load depth map
    cv::Mat depthMap; // The values are saved in decimili-meters which is 10e-4 meters
    load_image("../../data/depth-input/" + idName + ".png", depthMap, cv::IMREAD_UNCHANGED);

    // Find depth of the largest green object
    int depth = depthMap.at<int16_t>(centers[circleId].x, centers[circleId].y);

    // Calculate the 3D position of the largest green object
    double depthd = depth * 1e-4;
    cv::Point3d point3D = pixel2World(centers[circleId], cameraIntrinsics, depthd);

    // Print the 3D position of the largest green object
    std::cout << "3D position of the largest green object: " << point3D << std::endl;

    // Surface normal of the largest green object
    // TODO: MAKE THIS


    // show the image and depth map
    cv::imshow("Depth map", depthMap);
    cv::imshow("Display window", prediction);
    cv::waitKey(0);

    return 0;
}