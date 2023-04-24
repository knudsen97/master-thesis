#include "../inc/CameraCalibration.hpp"


CameraCalibration::CameraCalibration(const cv::Size &grid_size, const cv::Size &image_size, int square_size)
{
    this->grid_size = grid_size;
    this->image_size = image_size;

    // Define a grid of object points with a square size of 19mm in a 13x9 grid
    double square_size_meters = (double)square_size / 1000;
    for (int i = 0; i < this->grid_size.height; i++)
    {
        for (int j = 0; j < this->grid_size.width; j++)
        {
            this->object_points.push_back(cv::Point3f(j * square_size_meters, i * square_size_meters, 0));
        }
    }

}

CameraCalibration::~CameraCalibration(){}

// Get images
void CameraCalibration::getImages(Sensor &sensor, int num_images)
{
    std::cout << "Getting images for camera calibration\n";
    std::cout << "Press 's' to save image\n";
    std::cout << "To skip image acquisition press 'q' or 'esc'" << std::endl;

    open3d::t::geometry::Image open3d_image, open3d_depth;
    std::vector<cv::Point2f> corners;
    cv::Mat image;

    int i = 0;
    while(true)
    {
        auto key = cv::waitKey(1);
        sensor.grabFrame(open3d_image, open3d_depth);
        sensor.open3d_to_cv(open3d_image, image, true);

        // Save image if 's' is pressed and chessboard is found
        if (key == 's')
        {
            bool chessboard_found = cv::findChessboardCorners(image, this->grid_size, corners);
            if (chessboard_found)
            {
                // Convert image to gray and save it
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                cv::imwrite("../calib_images/image" + std::to_string(i) + ".png", image);
                i++;
                std::cout << "Image " << i << " saved" << std::endl;
            }
            else
            {
                std::cout << "Chessboard not found. Try again..." << std::endl;
                continue;
            }
        }
        else if (key == 'q' || key == 27)
        {
            std::cout << "Calibration stopped... Exiting program" << std::endl;
            break;
            return;
        }
        else if (i >= num_images)
        {
            std::cout << "Number of images reached... Exiting program" << std::endl;
            break;
        }

        cv::imshow("Image", image);
    }

    // Close all windows
    cv::destroyAllWindows();
}

// Calibrate
/**
 * @brief Calibrate the camera.
 * @param square_size Size of the square in millimeters
 * @details This function calibrates the camera using the images saved in the calib_images folder. Using the guide from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
 */
cv::Mat CameraCalibration::calibrate(double square_size)
{
    // Termination criteria
    auto criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    // Create a vector to store the object points and image points from all the images
    std::vector<std::vector<cv::Point3f>> obj_points_vec;
    std::vector<std::vector<cv::Point2f>> img_points_vec;

    // Get the images
    std::vector<cv::String> filenames;
    cv::glob("../calib_images/*.png", filenames);

    // Loop through the images
    std::cout << "Finding chessboard corners in images..." << std::endl;
    for (auto filename : filenames)
    {
        // Read the image
        cv::Mat image = cv::imread(filename);

        // Convert to gray
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Find the chessboard corners
        std::vector<cv::Point2f> corners;
        bool chessboard_found = cv::findChessboardCorners(gray, this->grid_size, corners);//, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        // If corners are found, add object points, image points (after refining them)
        if (chessboard_found)
        {
            obj_points_vec.push_back(this->object_points);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            img_points_vec.push_back(corners);

            // Draw and display the corners
            cv::drawChessboardCorners(image, this->grid_size, corners, chessboard_found);
            cv::imshow("Image", image);
            cv::waitKey(100);
        }
    }

    // Calibrate the camera
    std::cout << "Calibrating camera..." << std::endl;
    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    cv::calibrateCamera(obj_points_vec, img_points_vec, this->image_size, camera_matrix, dist_coeffs, rvecs, tvecs);

    cv::Mat rmat;
    cv::Rodrigues(rvecs[0], rmat);
    cv::Mat extrinsics(3, 4, rmat.type());
    rmat.copyTo(extrinsics(cv::Rect(0, 0, 3, 3)));
    tvecs[0].copyTo(extrinsics(cv::Rect(3, 0, 1, 3)));
    std::cout << "Extrinsics: \n" << extrinsics << std::endl;

    // Save the camera matrix and distortion coefficients as yaml file
    std::cout << "Saving camera matrix and distortion coefficients..." << std::endl;

    cv::FileStorage fs("../config/calibration.yaml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs.release();


    // Calculate the reprojection error
    std::cout << "Reprojection error" << std::endl;
    double total_error = 0;
    for (size_t i = 0; i < obj_points_vec.size(); i++)
    {
        std::vector<cv::Point2f> img_points_reproj;
        cv::projectPoints(obj_points_vec[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs, img_points_reproj);
        double error = cv::norm(img_points_vec[i], img_points_reproj, cv::NORM_L2) / img_points_reproj.size();
        total_error += error;
    }
    std::cout << "Total error: " << total_error / obj_points_vec.size() << std::endl;


    // Show an example of undistortion
    cv::Mat image = cv::imread("../calib_images/image0.png");
    cv::Mat undistorted;
    cv::undistort(image, undistorted, camera_matrix, dist_coeffs);

    cv::imshow("Image", image);
    cv::imshow("Undistorted", undistorted);
    cv::waitKey(0);

    // Close all windows
    cv::destroyAllWindows();

    std::cout << "Camera calibration done!" << std::endl;
    
    return camera_matrix;
}

cv::Mat CameraCalibration::calculateExtrinsics(Sensor &sensor, const std::string &filename)
{
    std::cout << "\nGetting extrinsics of the camera...\n";
    std::cout << "Press 's' to save the extrinsics" << std::endl;
    std::cout << "Press 'q' or 'esc' to skip extrinsic step and use previous." << std::endl;

    // Read camera matrix and distortion coefficients from yaml file
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeffs;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();

    // Create open3d and cv images/mats
    open3d::t::geometry::Image open3d_image, open3d_depth;
    cv::Mat image, gray;
    cv::Mat rvec, tvec;
    cv::Mat extrinsics;

    // Start extrinsic estimation
    while (true)
    {
        auto key = cv::waitKey(1);
        sensor.grabFrame(open3d_image, open3d_depth);
        sensor.open3d_to_cv(open3d_image, image, true);

        if (key =='s')
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            // Find the chessboard corners
            std::vector<cv::Point2f> img_points;
            bool chessboard_found = cv::findChessboardCorners(gray, this->grid_size, img_points);//, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

            // If corners are found, add object points, image points (after refining them)
            if (chessboard_found)
            {
                cv::cornerSubPix(gray, img_points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
                cv::solvePnP(this->object_points, img_points, camera_matrix, dist_coeffs, rvec, tvec);

                cv::Mat rmat;
                cv::Rodrigues(rvec, rmat);
                extrinsics = cv::Mat(3, 4, rmat.type());
                rmat.copyTo(extrinsics(cv::Rect(0, 0, 3, 3)));
                tvec.copyTo(extrinsics(cv::Rect(3, 0, 1, 3)));
            }
            else
            {
                std::cout << "Chessboard not found!" << std::endl;
                continue;
            }

            if (!extrinsics.empty())
            {
                // Save extrinsics to a YAML file
                fs.open("../config/extrinsics.yaml", cv::FileStorage::WRITE);
                fs << "extrinsics" << extrinsics;
                fs.release();
                break;
            }
        }
        else if (key == 'q' || key == 27)
            break;

        cv::imshow("Estimating Extrinsics", image);
    }
    



    

    

    return extrinsics;
}
