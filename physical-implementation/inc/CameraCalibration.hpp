#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <opencv2/opencv.hpp>
#include "../inc/Sensor.hpp"
#include <open3d/Open3D.h>



class CameraCalibration
{

    public:
    CameraCalibration(const cv::Size &grid_size, const cv::Size &image_size, int square_size);
    ~CameraCalibration();

    // Get images
    void getImages(Sensor &sensor, int num_images);

    // Calibrate
    cv::Mat calibrate(double square_size);

    // Calculate extrinsics
    cv::Mat calculateExtrinsics(Sensor &sensor, const std::string &filename_calibration, const std::string &filename_extrinsics);//, cv::Mat &rvecs, cv::Mat &tvecs);

    private:
    cv::Size grid_size;
    cv::Size image_size;

    std::vector<cv::Point3f> object_points;
};

#endif
