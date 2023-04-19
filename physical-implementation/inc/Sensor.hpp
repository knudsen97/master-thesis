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


class Sensor
{
private:
    open3d::t::io::RealSenseSensor rs;
    int width;
    int height;

    cv::Mat intrinsics;
    Eigen::Matrix4d extrinsics;

public:
    // Default constructor
    Sensor();

    // Constructor
    Sensor(const std::string& config_filename);

    // Intrinsics
    void setIntrinsics(double fovy);
    void getIntrinsics(cv::Mat& intrinsics);

    // Extrinsics
    void setExtrinsics(const Eigen::Matrix4d& extrinsics);
    void getExtrinsics(Eigen::Matrix4d& extrinsics);

    std::pair<int,int> getResolution();

    void createPinholeCameraIntrinsics(open3d::camera::PinholeCameraIntrinsic &camera_intrinsics);

    // Start and stop capture
    void startCapture();
    void stopCapture();

    // Grab frame
    void grabFrame(open3d::t::geometry::Image &image, open3d::t::geometry::Image &depth);

    // Default destructor
    ~Sensor();
};

#endif // SENSOR_H