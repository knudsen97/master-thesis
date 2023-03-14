#include "../inc/PredictionProcessor.hpp"

// ------------------- Constructors and destructor ------------------- 
PredictionProcessor::PredictionProcessor()
{
    std::cout << "Default PredictionProcessor constructor called." << std::endl;
    std::cout << "Using depth scale of 10000";
    this->_depth_scale = 10000;
    this->_flip_mat << 1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, -1, 0,
                        0, 0, 0, 1;
}

PredictionProcessor::PredictionProcessor(double depth_scale) :
    _depth_scale(depth_scale)
{
    this->_flip_mat << 1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, -1, 0,
                        0, 0, 0, 1;
}

PredictionProcessor::~PredictionProcessor(){}


// ------------------- Public Methods -------------------
void PredictionProcessor::setIntrinsicsAndExtrinsics(const open3d::camera::PinholeCameraIntrinsic &intrinsics, const Eigen::Matrix4d &extrinsics)
{
    this->_intrinsics = intrinsics;
    this->_extrinsics = extrinsics;
}

void PredictionProcessor::setFlipMatrix(const Eigen::Matrix4d &flip_mat)
{
    this->_flip_mat = flip_mat;
}


void PredictionProcessor::createPCFromDepth(const cv::Mat &depth_in, PointCloudPtr &pc_out, bool flip)
{
    open3d::geometry::Image depth_image;
    this->cv2o3dImage(depth_in, depth_image);

    pc_out = open3d::geometry::PointCloud::CreateFromDepthImage(depth_image, this->_intrinsics, this->_extrinsics, this->_depth_scale);
    if (flip)
        pc_out->Transform(this->_flip_mat);
}

void PredictionProcessor::estimateAllNormals(PointCloudPtr &pc, double radius, int nn_max, bool normalize)
{
    const open3d::geometry::KDTreeSearchParamHybrid search_param(radius, nn_max);
    pc->EstimateNormals(search_param);
    if (normalize)
        pc->NormalizeNormals();
}

void PredictionProcessor::pixel2cam(const cv::Mat &depth, const cv::Point2d &p, cv::Point3d &p_cam)
{
    auto f = this->_intrinsics.GetFocalLength();
    auto c = this->_intrinsics.GetPrincipalPoint();

    p_cam.z = depth.at<uint16_t>(cv::Point(p.x, p.y)) / this->_depth_scale;
    p_cam.x = (p.x - c.first)  * p_cam.z / f.first;
    p_cam.y = (p.y - c.second) * p_cam.z / f.second;
}


int PredictionProcessor::findIndexOfClosestPoint(const PointCloudPtr &pc, const cv::Point3d &p_cam, bool flip)
{
    auto pc_center_3d = open3d::geometry::PointCloud();
    pc_center_3d.points_.push_back(Eigen::Vector3d(p_cam.x, p_cam.y, p_cam.z));
    if (flip)
        pc_center_3d.Transform(this->_flip_mat);
    auto distances = pc->ComputePointCloudDistance(pc_center_3d);

    // Find index of closest point in distances
    int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
    return min_index;
}


void PredictionProcessor::computeRotationMatrixFromNormal(const Eigen::Vector3d &normal, cv::Mat &R)
{
    Eigen::Vector3d u;
    if (normal[0] != 0 || normal[1] != 0)
        u = Eigen::Vector3d(1, 0, 0);
    else
        u = Eigen::Vector3d(0, 1, 0);

    Eigen::Vector3d v = normal.cross(u);
    u = v.cross(normal);

    u = u / u.norm();
    v = v / v.norm();
    Eigen::Vector3d n = normal / normal.norm();

    R = cv::Mat::zeros(3, 3, CV_64F);
    for (size_t i = 0; i < 3; i++)
    {
        R.at<double>(0, i) = u[i];
        R.at<double>(1, i) = v[i];
        R.at<double>(2, i) = n[i];
    }
}



// ------------------- Private Methods -------------------
void PredictionProcessor::cv2o3dImage(const cv::Mat &cv_image_in, open3d::geometry::Image &o3d_image_out)
{
// open3d::geometry::Image depth_image;
    const int width = cv_image_in.cols;
    const int height = cv_image_in.rows;
    o3d_image_out.Prepare(width, height, 1, sizeof(uint16_t));
    const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(cv_image_in.data);
    std::memcpy(o3d_image_out.data_.data(), data_ptr, width * height * sizeof(uint16_t));
}


