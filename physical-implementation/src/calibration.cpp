#include "../inc/CameraCalibration.hpp"


int main(int argc, char* argv[])
{
    int num_images = 10;

    for (size_t i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--num_images")
        {
            num_images = std::stoi(argv[i + 1]);
        }
    }
    std::cout << "num_images: " << num_images << std::endl;

    // Load camera configuration into a Sensor object
    const std::string config_filename = "../config/camera_config.json";
    Sensor sensor(config_filename);
    
    // Start capture
    sensor.startCapture();


    // CAMERA CALIBRATION
    cv::Size board_size(13, 9);
    cv::Size image_size(640, 480);
    int square_size = 19;
    // int num_images = 10;

    CameraCalibration calib(board_size, image_size, square_size);
    calib.getImages(sensor, num_images);
    cv::Mat intrinsics = calib.calibrate(square_size);

    // Stop capture
    sensor.stopCapture();
    

    return 0;
}