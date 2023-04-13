#ifndef SENSOR_H
#define SENSOR_H

#include <rw/core/Ptr.hpp>
#include <rw/kinematics/State.hpp>
#include <rwlibs/simulation/SimulatedCamera.hpp>
#include <rwlibs/simulation/SimulatedScanner25D.hpp>
#include <rw/geometry/PointCloud.hpp>


using namespace rw::core;
using namespace rw::common;
using namespace rw::kinematics;
using rw::sensor::Image;
using namespace rwlibs::simulation;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/rgbd/linemod.hpp>

#include <open3d/Open3D.h>

typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;

enum ImageType
{
    BGR,
    RGB
};

class SimulatedRGBD
{
private:
    SimulatedCamera::Ptr sim_cam;
    SimulatedScanner25D::Ptr sim_depth;
    cv::Mat intrinsics;

public:
    // Default constructor
    SimulatedRGBD();

    // Constructor
    SimulatedRGBD(const SimulatedCamera &sim_cam, const SimulatedScanner25D &sim_depth, const cv::Mat &intrinsics = cv::Mat());

    void convertImageToCV(const Image *image, cv::Mat &cv_image, int imgType = ImageType::BGR);

    void initCamera(int fps);
    void initScanner25D(int fps);

    void acquireImage(State &state, const Simulator::UpdateInfo &info);
    void acquireDepth(State &state, const Simulator::UpdateInfo &info);

    void addDepthNoise(cv::Mat &depth, double mean=0.0, double stddev=0.1, double scale = 1.0);

    void getImage(cv::Mat &image_out, int imgType = ImageType::BGR);
    void getPointCloudAndDepthImage(PointCloudPtr &pc_out, cv::Mat &depthImage_out);//, const cv::Mat &R, const cv::Mat &t);


    void close();

    // Destructor
    ~SimulatedRGBD();
};


#endif // SENSOR_H