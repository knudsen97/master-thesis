#include "../inc/Sensor.hpp"

Sensor::Sensor()
{
    std::cout << "Standard Sensor constructor called" << std::endl;
    std::cout << "Using default configuration" << std::endl;
    const std::string config_filename = "../config/camera_config.json";
    
    // Load configuration file
    open3d::t::io::RealSenseSensorConfig rs_cfg;
    open3d::io::ReadIJsonConvertible(config_filename, rs_cfg);

    // Read configuration file and extract width and height
    auto res = rs_cfg.config_["color_resolution"];
    this->width = std::stoi(res.substr(0, res.find(",")));
    this->height = std::stoi(res.substr(res.find(",") + 1, res.length()));

    // Initialize sensor
    open3d::t::io::RealSenseSensor rs;
    rs.InitSensor(rs_cfg);

    // Start capture
    rs.StartCapture(true);
}

Sensor::Sensor(const std::string& config_filename)
{
    std::cout << "Sensor constructor called" << std::endl;
    std::cout << "Using configuration file: " << config_filename << std::endl;
    
    // Load configuration file
    open3d::t::io::RealSenseSensorConfig rs_cfg;
    open3d::io::ReadIJsonConvertible(config_filename, rs_cfg);

    // Read configuration file and extract width and height
    auto res = rs_cfg.config_["color_resolution"];
    this->width = std::stoi(res.substr(0, res.find(",")));
    this->height = std::stoi(res.substr(res.find(",") + 1, res.length()));

    // Initialize sensor
    this->rs.InitSensor(rs_cfg);
}

void Sensor::setIntrinsics(double fovy)
{
    double fovy_pixel = this->height / 2 / tan (fovy * (2 * M_PI) / 360.0 / 2.0);

    this->intrinsics = (cv::Mat_<double>(3, 3) << fovy_pixel, 0.0,        this->width/2.0, 
                                                  0.0,        fovy_pixel, this->height/2.0, 
                                                  0.0,        0.0,        1.0);
    std::cout << "Intrinsics from Sensor class: \n" << this->intrinsics << std::endl;
}

void Sensor::setIntrinsics(const cv::Mat &intrinsics)
{
    this->intrinsics = intrinsics;
}


void Sensor::getIntrinsics(cv::Mat &intrinsics)
{
    this->intrinsics.copyTo(intrinsics);
}

void Sensor::setExtrinsics(const Eigen::Matrix4d& extrinsics)
{
    this->extrinsics = extrinsics;
}

void Sensor::setExtrinsics(const cv::Mat& extrinsics)
{
    // Convert cv::Mat to Eigen::Matrix4d
    Eigen::Matrix4d extrinsics_eigen;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            extrinsics_eigen(i, j) = extrinsics.at<double>(i, j);
        }
    }

    this->extrinsics = extrinsics_eigen;
}


void Sensor::getExtrinsics(Eigen::Matrix4d& extrinsics)
{
    extrinsics = this->extrinsics;
}

std::pair<int,int> Sensor::getResolution()
{
    return std::make_pair(this->width, this->height);
}


void Sensor::createPinholeCameraIntrinsics(open3d::camera::PinholeCameraIntrinsic &camera_intrinsics)
{
    camera_intrinsics = open3d::camera::PinholeCameraIntrinsic(
                            this->width, 
                            this->height, 
                            this->intrinsics.at<double>(0, 0), 
                            this->intrinsics.at<double>(1, 1), 
                            this->intrinsics.at<double>(0, 2), 
                            this->intrinsics.at<double>(1, 2)
                            );
}



void Sensor::startCapture()
{
    rs.StartCapture(true);
}

void Sensor::stopCapture()
{
    rs.StopCapture();
}

void Sensor::grabFrame(open3d::t::geometry::Image &image, open3d::t::geometry::Image &depth)
{

    auto im_rgbd = rs.CaptureFrame(true, true);  // wait for frames and align them
        
    // Grab color and depth images
    image = im_rgbd.color_;
    depth = im_rgbd.depth_;
    // auto open3d_image = im_rgbd.color_;
    // auto open3d_depth = im_rgbd.depth_;
}

void Sensor::open3d_to_cv(open3d::t::geometry::Image &open3d_image, cv::Mat &cv_image, bool is_color)
{
    if (is_color)
    {
        cv::Mat cv_image_copy(open3d_image.GetRows(), open3d_image.GetCols(),
                    CV_8UC(open3d_image.GetChannels()),
                    open3d_image.GetDataPtr());
        cv::cvtColor(cv_image_copy, cv_image_copy, cv::COLOR_RGB2BGR);
        cv_image_copy.copyTo(cv_image);
    }
    else
    {
        cv::Mat cv_image_copy(open3d_image.GetRows(), open3d_image.GetCols(),
                    CV_16UC(open3d_image.GetChannels()),
                    open3d_image.GetDataPtr());
        cv_image_copy.copyTo(cv_image);
    }
}




Sensor::~Sensor()
{}