#include "../inc/Sensor.hpp"

// Default constructor
SimulatedRGBD::SimulatedRGBD(){}

// Constructor
SimulatedRGBD::SimulatedRGBD(const SimulatedCamera &sim_cam, const SimulatedScanner25D &sim_depth) 
    : sim_cam( ownedPtr (new SimulatedCamera(sim_cam)) ),
        sim_depth( ownedPtr (new SimulatedScanner25D(sim_depth)) )
{
    this->sim_depth->open();
}

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


void SimulatedRGBD::initCamera(int fps)
{
    this->sim_cam->setFrameRate(fps);
    this->sim_cam->initialize();
    this->sim_cam->start();
}

void SimulatedRGBD::initScanner25D(int fps)
{
    this->sim_depth->setFrameRate(fps);
    this->sim_depth->open();
}

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


void SimulatedRGBD::getImage(cv::Mat &image_out, int imgType)
{
    const Image *image = this->sim_cam->getImage();
    SimulatedRGBD::convertImageToCV(image, image_out, imgType);
}

void SimulatedRGBD::getDepth(std::shared_ptr<open3d::geometry::PointCloud> &pc_out)
{
    const rw::geometry::PointCloud::Ptr pc = ownedPtr( new rw::geometry::PointCloud(this->sim_depth->getScan()) );
    std::vector<rw::math::Vector3Df> data = pc->getData();

    // Convert Vector3Df to eigen vector and Ignore points with -10 depth, because that is just the default value (background)
    std::vector<Eigen::Vector3d> data_eigen;
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i][2] != -10)
            data_eigen.push_back(Eigen::Vector3d(data[i][0], data[i][1], data[i][2]));
    } 

    for (size_t i = 0; i < data_eigen.size(); i++)
    {
        std::cout << data_eigen[i][0] << ", " << data_eigen[i][1] << ", " << data_eigen[i][2] << std::endl;

        if (i>10)
            break;
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



    // Create Open3D visualizer
    // open3d::visualization::VisualizerWithKeyCallback o3d_vis;
    // o3d_vis.CreateVisualizerWindow("Open3D", 640, 480);
    // o3d_vis.AddGeometry(o3d_pc);
    // o3d_vis.Run();

    // o3d_vis.DestroyVisualizerWindow();
}

void SimulatedRGBD::close()
{
    this->sim_cam->stop();
    this->sim_depth->close();
}

SimulatedRGBD::~SimulatedRGBD()
{
    // this->sim_cam->stop();
    // this->sim_depth->close();
}