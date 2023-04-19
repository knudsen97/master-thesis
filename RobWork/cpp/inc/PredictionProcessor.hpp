#ifndef PREDICTIONPROCESSOR_H
#define PREDICTIONPROCESSOR_H

// Include Open3D headers
#include <open3d/Open3D.h>

// Include OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

//include RobWork headers
#include <rw/math.hpp>

typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;

class PredictionProcessor
{
private:
    /* data */
    double _depth_scale;
    open3d::camera::PinholeCameraIntrinsic _intrinsics;
    Eigen::Matrix4d _extrinsics;
    Eigen::Matrix4d _flip_mat;

public:
    // Constructors and destructor
    PredictionProcessor();
    PredictionProcessor(double depth_scale);
    ~PredictionProcessor();

    // Methods
    void setIntrinsicsAndExtrinsics(const open3d::camera::PinholeCameraIntrinsic &intrinsics, const Eigen::Matrix4d &extrinsics);
    void setFlipMatrix(const Eigen::Matrix4d &flip_mat);
    void createPCFromDepth(const cv::Mat &depth_in, PointCloudPtr &pc_out, bool flip=true);
    void estimateAllNormals(PointCloudPtr &pc, double radius, int nn_max, bool normalize=true);
    void pixel2cam(const cv::Mat &depth, const cv::Point2d &p, cv::Point3d &p_cam);
    int findIndexOfClosestPoint(const PointCloudPtr &pc, const cv::Point3d &p_cam, bool flip=true);

    void computeRotationMatrixFromNormal(const Eigen::Vector3d &normal, rw::math::Rotation3D<double> &R);
    void computeCenters(const cv::Mat &image, std::vector<cv::Point2i> &dest, int max_radius=10000, int min_radius = 0);


private:
    void cv2o3dImage(const cv::Mat &cv_image_in, open3d::geometry::Image &o3d_image_out);

};




#endif