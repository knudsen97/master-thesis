// Standard C++ includes
#include <iostream>

// Include OpenCV header file
#include <opencv2/opencv.hpp>

// Include Open3D header file
#include <open3d/Open3D.h>

// Include header files
#include "../inc/Sensor.hpp"
#include "../inc/PredictionProcessor.hpp"
#include "../inc/Inference.hpp"


typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;


void open3d_to_cv(open3d::t::geometry::Image& open3d_image, cv::Mat& cv_image, bool is_color = true)
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


int main(int argc, char* argv[])
{
    // Check if the camera is connected
    if (!open3d::t::io::RealSenseSensor::ListDevices())
    {
        std::cout << "No camera detected" << std::endl;
        return 0;
    }

    std::string model_file_name = "unet_resnet101_1_jit.pt";
    std::string model_name;

    for (size_t i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model")
        {
            model_file_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--model_name")
        {
            model_name = argv[i + 1];
        }
    }
    if (model_name.empty())
    {
        model_name = model_file_name.substr(0, model_file_name.size() - 7);
    }

    // Load camera configuration into a Sensor object
    const std::string config_filename = "../config/camera_config.json";
    Sensor sensor(config_filename);

    // Create intrinsics and extrinsics matrices
    Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();
    cv::Mat intrinsics;
    open3d::camera::PinholeCameraIntrinsic camera_intrinsics;

    // Set intrinsics and extrinsics
    sensor.setIntrinsics(50.0);
    sensor.setExtrinsics(extrinsics);

    // Create flip matrix to flip the point cloud
    Eigen::Matrix4d flip_mat;
    flip_mat << 1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, 1;
    
    // Create PredictionProcessor object
    double depth_scale = 1e4;
    PredictionProcessor processor(depth_scale);

    cv::Mat image, pred_image, depth;
    cv::Mat returned_image;
    bool inference_sucess;
    Inference inf("../../jit_models/" + model_file_name);

    // Start capture
    sensor.startCapture();

    while(true)
    {
        auto key = cv::waitKey(1);

        // Grab frame
        open3d::t::geometry::Image open3d_image, open3d_depth;
        sensor.grabFrame(open3d_image, open3d_depth);

        // Convert Open3D image to OpenCV image
        open3d_to_cv(open3d_image, image, true);
        open3d_to_cv(open3d_depth, depth, false);

        // Create point cloud
        auto open3d_depth_legacy = open3d_depth.ToLegacy();
        sensor.createPinholeCameraIntrinsics(camera_intrinsics);
        sensor.getExtrinsics(extrinsics);
        int depth_scale = 1000;
        PointCloudPtr pc;
        pc = open3d::geometry::PointCloud::CreateFromDepthImage(open3d_depth_legacy, camera_intrinsics, extrinsics, depth_scale);
        pc->Transform(flip_mat);

        // Do prediction
        if (key == 'k')
        {
            image.copyTo(pred_image);
            auto time_start = std::chrono::high_resolution_clock::now();
            inference_sucess = inf.predict(pred_image, returned_image);
            auto time_end = std::chrono::high_resolution_clock::now();
            std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
    
            cv::Point center = cv::Point(400, 200);
            std::vector<cv::Point> centers;
            processor.computeCenters(returned_image, centers, 10000);

            if (centers.size() > 0)
            {
                center = centers[0];
                for (auto c : centers)
                    std::cout << "center: " << c << std::endl;
                cv::circle(pred_image, center, 5, cv::Scalar(0, 0, 255), -1);
                cv::circle(returned_image, center, 5, cv::Scalar(0, 0, 0), -1);
            }
            else
            {
                std::cout << "No graspable areas found" << std::endl;
            }

            if (inference_sucess)
            {
                std::cout << "Inference success" << std::endl;
                cv::imshow("Image", pred_image);
                cv::imshow("Prediction", returned_image);
            }
            else
                std::cout << "Inference failure" << std::endl;
        }
        
        //

        // Visualization
        cv::imshow("color", image);
        cv::imshow("depth", depth);

        // If you want to create point cloud press 'p' or close the program with 'q' or 'esc'
        if (key == 'p')
        {
            auto resolution = sensor.getResolution();
            open3d::visualization::VisualizerWithKeyCallback o3d_vis;
            o3d_vis.CreateVisualizerWindow("PointCloud", resolution.first, resolution.second);
            o3d_vis.AddGeometry(pc);
            o3d_vis.Run();
        }
        else if (key == 'q' || key == 27)
        {
            break;
        }
    }

    sensor.stopCapture();
  
    return 0;
}
