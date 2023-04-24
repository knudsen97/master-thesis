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
#include "../inc/CameraCalibration.hpp"


typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;


void load_intrinsics(const std::string &filename, cv::Mat &intrinsics_out, cv::Mat &dist_coeff_out);
void load_extrinsics(const std::string &filename, cv::Mat &extrinsics);

int main(int argc, char* argv[])
{
    // Check if the camera is connected
    if (!open3d::t::io::RealSenseSensor::ListDevices())
    {
        std::cout << "No camera detected" << std::endl;
        return 0;
    }


    //-------------------------
    std::string model_file_name = "unet_resnet101_1_jit.pt";
    std::string model_name;
    bool find_extrinsics = false;

    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model")
        {
            model_file_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--model_name")
        {
            model_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--find_extrinsics")
        {
            std::string arg = argv[i + 1];
            // Change string to lowercase
            std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c){ return std::tolower(c); });
            std::cout << "arg: " << arg << std::endl;
            if (arg == "true")
                find_extrinsics = true;
        }
    }
    if (model_name.empty())
    {
        model_name = model_file_name.substr(0, model_file_name.size() - 7);
    }

    // Create PredictionProcessor object
    double depth_scale = 1e4;
    PredictionProcessor processor(depth_scale);

    // Create Inference object
    cv::Mat image, pred_image, depth;
    cv::Mat returned_image;
    bool inference_sucess;
    Inference inf("../../models/" + model_file_name);

    // Load camera configuration into a Sensor object
    const std::string config_filename = "../config/camera_config.json";
    Sensor sensor(config_filename);
    
    // Start capture
    sensor.startCapture();

    // Create intrinsics and extrinsics matrices
    Eigen::Matrix4d extrinsics_eigen = Eigen::Matrix4d::Identity();
    open3d::camera::PinholeCameraIntrinsic camera_intrinsics;
    cv::Mat intrinsics, extrinsics, dist_coeff;

    // CAMERA CALIBRATION (EXTRINSICS)
    cv::Size board_size(13, 9);
    cv::Size image_size(640, 480);
    int square_size = 19;
    CameraCalibration calib(board_size, image_size, square_size);

    // Load intrinsics and extrinsics from YAML files
    cv::FileStorage fs;
    load_intrinsics("../config/calibration.yaml", intrinsics, dist_coeff);
    if(!intrinsics.empty())
        sensor.setIntrinsics(intrinsics);
    else // use default intrinsics
        sensor.setIntrinsics(50);

    // Estimate extrinsics of the camera or load from YAML file
    if (find_extrinsics)
        extrinsics = calib.calculateExtrinsics(sensor, "../config/calibration.yaml");
    else // Load extrinsics from YAML file
        load_extrinsics("../config/extrinsics.yaml", extrinsics);

    // sensor.setExtrinsics(extrinsics); // This does not work if the pointcloud
    sensor.setExtrinsics(extrinsics_eigen); // Has to be identity matrix for pointcloud

    // Close all cv windows
    cv::destroyAllWindows();

    // Print intrinsic and extrinsics found/loaded
    std::cout << "Intrinsics:\n" << intrinsics << std::endl;
    std::cout << "Extrinsics:\n" << extrinsics << std::endl;

    // Create flip matrix to flip the point cloud
    Eigen::Matrix4d flip_mat;
    flip_mat << 1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, 1;

    while(true)
    {
        auto key = cv::waitKey(1);

        // Grab frame
        open3d::t::geometry::Image open3d_image, open3d_depth;
        sensor.grabFrame(open3d_image, open3d_depth);

        // Convert Open3D image to OpenCV image
        sensor.open3d_to_cv(open3d_image, image, true);
        sensor.open3d_to_cv(open3d_depth, depth, false);

        // Create point cloud
        auto open3d_depth_legacy = open3d_depth.ToLegacy();
        sensor.createPinholeCameraIntrinsics(camera_intrinsics);
        sensor.getExtrinsics(extrinsics_eigen);
        int depth_scale = 1000;
        PointCloudPtr pc;
        pc = open3d::geometry::PointCloud::CreateFromDepthImage(open3d_depth_legacy, camera_intrinsics, extrinsics_eigen, depth_scale);
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

        // Visualization
        cv::imshow("depth", depth);
        cv::imshow("color", image);

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


void load_intrinsics(const std::string &filename, cv::Mat &intrinsics_out, cv::Mat &dist_coeff_out)
{
    cv::FileStorage fs("../config/calibration.yaml", cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["camera_matrix"] >> intrinsics_out;
        fs["distortion_coefficients"] >> dist_coeff_out;
        fs.release();
    }
    else
    {
        std::cout << "No intrinsics file found. Using default intrinsics or run the calibration script." << std::endl;
    }
}

void load_extrinsics(const std::string &filename, cv::Mat &extrinsics)
{
    cv::FileStorage fs("../config/extrinsics.yaml", cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["extrinsics"] >> extrinsics;
        fs.release();
    }
    else
    {
        std::cout << "No extrinsics file found. Using default extrinsics." << std::endl;
    }
}