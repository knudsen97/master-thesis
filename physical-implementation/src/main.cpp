#include <iostream>

// Include OpenCV header file
#include <opencv2/opencv.hpp>

// Include Intel Realsense Cross Platform API
#include <librealsense2/rs.hpp>



int main(int argc, char* argv[])
{
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg; 
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Start streaming with default configuration
    pipe.start(cfg);

    // Load the intrinsic parameters of the camera
    rs2::pipeline_profile profile = pipe.get_active_profile();
    rs2::video_stream_profile depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = depth_stream.get_intrinsics();

    // Get the camera's field of view
    double fovy = 2 * atan2(intrinsics.height, 2 * intrinsics.fx) * 180 / M_PI;

    std::cout << "Camera's field of view: " << fovy << std::endl;


    while (cv::waitKey(1) < 0)
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        // Get the depth and color frames
        auto depth_frame = frames.get_depth_frame();
        auto color_frame = frames.get_color_frame();

        auto height = depth_frame.get_height();
        auto width = depth_frame.get_width();

        // Convert the depth frame to a grayscale image
        cv::Mat depth_image(cv::Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat image(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // Get the depth frame's dimensions
        double fovy_pixel = height / 2 / tan (fovy * (2 * M_PI) / 360.0 / 2.0);

        std::cout << "Camera's field of view: " << fovy_pixel << "\r";



        // Show the depth image
        cv::imshow("Depth Image", depth_image);
        cv::imshow("Image", image);

    }

    // Stop streaming
    pipe.stop();

    return 0;
}


        // depth_frame depth = frames.get_depth_frame();
        // Mat depthImage(Size(depth.get_width(), depth.get_height()), CV_16UC1, (void*)depth.get_data(), Mat::AUTO_STEP);