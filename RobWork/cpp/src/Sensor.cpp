#include "../inc/Sensor.hpp"

/**
 * @brief Default constructor for SimulatedRGBD object.
*/
SimulatedRGBD::SimulatedRGBD(){}

/**
 * @brief Constructor for SimulatedRGBD object.
 * @param sim_cam The simulated camera to use.
 * @param sim_depth The simulated depth scanner to use.
 * @param intrinsics The intrinsics of the camera. If no intrinsics are provided, the default intrinsics are used.
*/
SimulatedRGBD::SimulatedRGBD(const SimulatedCamera &sim_cam, const SimulatedScanner25D &sim_depth, const cv::Mat &intrinsics) 
    : sim_cam( ownedPtr (new SimulatedCamera(sim_cam)) ),
        sim_depth( ownedPtr (new SimulatedScanner25D(sim_depth)) )
{
    this->sim_depth->open();

    if (intrinsics.empty())
    {
        std::cout << "No intrinsics provided. Using default intrinsics." << std::endl;
        this->intrinsics = (cv::Mat_<double>(3, 3) << 430.0, 0.0,   320.0, 
                                                      0.0,   430.0, 240.0, 
                                                      0.0,   0.0,   1.0);
    }
    else
    {
        this->intrinsics = intrinsics;
    }
    // std::cout << "Intrinsics: \n" << this->intrinsics << std::endl;
}


/**
 * @brief Convert RobWork Image to OpenCV Mat
 * @param image Raw pointer to RobWork Image
 * @param cv_image OpenCV Mat to store the converted image
 * @param imgType Type of image to convert to. Use ImageType::RGB or ImageType::BGR
*/
void SimulatedRGBD::convertImageToCV(const Image *image, cv::Mat &cv_image, int imgType)
{
    cv::Mat _image = cv::Mat(image->getHeight(), image->getWidth(), CV_8UC3, (Image*)image->getImageData());
    // cv::Mat imflip, imflip_mat;
    cv::flip(_image, _image, 1);
    if (imgType == ImageType::RGB)
        _image = _image;
    else if (imgType == ImageType::BGR)
        cv::cvtColor(_image, _image, cv::COLOR_BGR2RGB);
    else
    {
        std::cout << "Image type not supported!" << std::endl;
        std::cout << "Use ImageType::RGB or ImageType::BGR. Image type set to BGR" << std::endl;
    }
    _image.copyTo(cv_image);
}


/**
 * @brief Initialize the simulated RGB camera
 * @param fps Frame rate of the simulated camera
*/
void SimulatedRGBD::initCamera(int fps)
{
    this->sim_cam->setFrameRate(fps);
    this->sim_cam->initialize();
    this->sim_cam->start();
}


/**
 * @brief Initialize the simulated depth scanner
 * @param fps Frame rate of the simulated scanner
*/
void SimulatedRGBD::initScanner25D(int fps)
{
    this->sim_depth->setFrameRate(fps);
    this->sim_depth->open();
}


/**
 * @brief Acquire image from simulated RGBD RealSense sensor. This does not get the image, but waits for the image to be ready.
 * @param state RobWork state of the workcell
 * @param info Simulator update info
*/
void SimulatedRGBD::acquireImage(State &state, const Simulator::UpdateInfo &info)
{
    std::cout << "Acquiring image from simulated RGBD Sensor..." << std::endl;
    int cnt = 0;
    std::cout << "Waiting for image to be ready. Iteration: ";
    
    while (!this->sim_cam->isImageReady())
    {
        std::cout << cnt << ", ";
        this->sim_cam->update(info, state);
        cnt++;
    }
    std::cout << "Image ready!" << std::endl;

}


/**
 * @brief Acquire depth image from simulated RGBD RealSense sensor. This does not get the depth image, but waits for the depth image to be ready.
 * @param state RobWork state of the workcell
 * @param info Simulator update info
*/
void SimulatedRGBD::acquireDepth(State &state, const Simulator::UpdateInfo &info)
{
    std::cout << "Acquiring depth image from simulated RGBD Sensor..." << std::endl;
    int cnt = 0;
    std::cout << "Waiting for depth image to be ready. Iteration: ";
    
    while (!this->sim_depth->isScanReady())
    {
        std::cout << cnt << ", ";
        this->sim_depth->update(info, state);
        cnt++;
    }
    std::cout << "Depth image ready!" << std::endl;
}

/**
 * @brief Get image from simulated RGBD RealSense sensor.
 * @param image_out Image in OpenCV format
 * @param imgType Image type. Use ImageType::RGB or ImageType::BGR
*/
void SimulatedRGBD::getImage(cv::Mat &image_out, int imgType)
{
    const Image *image = this->sim_cam->getImage();
    SimulatedRGBD::convertImageToCV(image, image_out, imgType);
}


/**
 * @brief Get point cloud and depth image from simulated RGBD RealSense sensor.
 * @param pc_out Point cloud pointer in Open3D format
 * @param depthImage_out Depth image in OpenCV format
*/
void SimulatedRGBD::getPointCloudAndDepthImage(PointCloudPtr &pc_out, cv::Mat &depthImage_out)//, const cv::Mat &R, const cv::Mat &t);
{
    const rw::geometry::PointCloud::Ptr pc = ownedPtr( new rw::geometry::PointCloud(this->sim_depth->getScan()) );
    std::vector<rw::math::Vector3Df> data = pc->getData();

    // Convert Vector3Df to eigen vector and Ignore points with -10 depth, because that is just the default value (background)
    std::vector<Eigen::Vector3d> data_eigen;
    std::vector<cv::Point3d> data_cv;

    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i][2] != -10)
            data_eigen.push_back(Eigen::Vector3d(data[i][0], data[i][1], data[i][2]));
        data_cv.push_back(cv::Point3d(data[i][0], data[i][1], data[i][2]));
    } 

    // Convert point cloud to Open3D point cloud
    pc_out = std::shared_ptr<open3d::geometry::PointCloud>(new open3d::geometry::PointCloud(data_eigen));

    // Flip the point cloud
    Eigen::Matrix4d flip_mat;
    flip_mat << -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

    pc_out->Transform(flip_mat);

    // Create depth image from point cloud
    cv::Mat depthImage = cv::Mat::zeros(480, 640, CV_16U);
    for (int x = 0; x < depthImage.cols; x++)
    {
        for (int y = 0; y < depthImage.rows; y++)
        {
            auto depth = pc->operator()(x, y)[2];
            int value = 0;
            if (depth != -10)
            {
                value = (int16_t) -10000 * depth;
            }
            
            depthImage.at<int16_t>(y, x) = value;
        }
        
    }
    cv::flip(depthImage, depthImage_out, 1);
}


/**
 * @brief Adds gaussian noise to the depth image.
 * @param depth The input depth image that gets noise added to it.
 * @param mean The mean used for noise generation.
 * @param stddev The standard deviation used for noise generation.
 * @param scale The scale used for noise generation.
*/
void SimulatedRGBD::addDepthNoise(cv::Mat &depth, double mean, double stddev, double scale)
{
    cv::Mat noise = cv::Mat::zeros(depth.size(), CV_64F);
    cv::randn(noise, mean, stddev);
    noise = noise * scale;

    for (int i = 0; i < noise.rows; i++)
    {
        for (int j = 0; j < noise.cols; j++)
        {
            // Scale noise and cast to uint16_t
            auto d = depth.at<uint16_t>(i, j);

            // If depth is 0, point is invalid
            if (d == 0)
                continue;

            // Add noise to depth
            int val = d + round(noise.at<double>(i, j));
            if (val < 0)
                val = 0;

            depth.at<uint16_t>(i, j) = val;
        }
    }
}


/**
 * @brief Closes RobWork simulated camera and depth sensor.
*/
void SimulatedRGBD::close()
{
    this->sim_cam->stop();
    this->sim_depth->close();
}


/**
 * @brief Destructor.
*/
SimulatedRGBD::~SimulatedRGBD()
{
    // this->sim_cam->stop();
    // this->sim_depth->close();
}