#include "../inc/PredictionProcessor.hpp"

// ------------------- Constructors and destructor ------------------- 
/**
 * @brief Default constructor for PredictionProcessor object.
*/
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

/**
 * @brief Constructor for PredictionProcessor object.
 * @param depth_scale The depth scale to use.
*/
PredictionProcessor::PredictionProcessor(double depth_scale) :
    _depth_scale(depth_scale)
{
    this->_flip_mat << 1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, -1, 0,
                        0, 0, 0, 1;
}

/**
 * @brief Destructor for PredictionProcessor object.
*/
PredictionProcessor::~PredictionProcessor(){}


// ------------------- Public Methods -------------------
/**
 * @brief Set the intrinsics and extrinsics of the used camera.
 * @param intrinsics The intrinsics.
 * @param extrinsics The extrinsics.
*/
void PredictionProcessor::setIntrinsicsAndExtrinsics(const open3d::camera::PinholeCameraIntrinsic &intrinsics, const Eigen::Matrix4d &extrinsics)
{
    this->_intrinsics = intrinsics;
    this->_extrinsics = extrinsics;
}

/**
 * @brief Set the flip matrix. The point cloud has to be flipped in order to be aligned with the RGB image.
 * @param flip_mat The flip matrix.
*/
void PredictionProcessor::setFlipMatrix(const Eigen::Matrix4d &flip_mat)
{
    this->_flip_mat = flip_mat;
}

/**
 * @brief Construct a point cloud from a depth image using Open3d.
 * @param depth_in The depth image in OpenCV format.
 * @param pc_out Point cloud to save the constructed point cloud to.
 * @param flip Whether to flip the point cloud or not.
*/
void PredictionProcessor::createPCFromDepth(const cv::Mat &depth_in, PointCloudPtr &pc_out, bool flip)
{
    open3d::geometry::Image depth_image;
    this->cv2o3dImage(depth_in, depth_image);

    pc_out = open3d::geometry::PointCloud::CreateFromDepthImage(depth_image, this->_intrinsics, this->_extrinsics, this->_depth_scale);
    if (flip)
        pc_out->Transform(this->_flip_mat);
}

/**
 * @brief Estimate the normals of a point cloud using Open3d.
 * @param pc The point cloud.
 * @param radius The radius to use for the normal estimation.
 * @param nn_max The maximum number of neighbors to use for the normal estimation.
 * @param normalize Whether to normalize the normals or not.
*/
void PredictionProcessor::estimateAllNormals(PointCloudPtr &pc, double radius, int nn_max, bool normalize)
{
    const open3d::geometry::KDTreeSearchParamHybrid search_param(radius, nn_max);
    pc->EstimateNormals(search_param);
    if (normalize)
        pc->NormalizeNormals();
}

/**
 * @brief Convert pixel coordinates to camera coordinates.
 * @param depth The depth image.
 * @param p The pixel coordinates.
 * @param p_cam The camera coordinates.
*/
void PredictionProcessor::pixel2cam(const cv::Mat &depth, const cv::Point2d &p, cv::Point3d &p_cam)
{
    auto f = this->_intrinsics.GetFocalLength();
    auto c = this->_intrinsics.GetPrincipalPoint();

    p_cam.z = depth.at<uint16_t>(cv::Point(p.x, p.y)) / this->_depth_scale;
    p_cam.x = (p.x - c.first)  * p_cam.z / f.first;
    p_cam.y = (p.y - c.second) * p_cam.z / f.second;
}


/**
 * @brief Find the index of the closest point in a point cloud to a given point.
 * @param pc The point cloud.
 * @param p_cam The point to find the closest point to.
 * @param flip Whether the point cloud is flipped or not.
*/
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

/**
 * @brief Compute the rotation matrix from a normal vector.
 * @param normal The normal vector.
 * @param R The rotation matrix.
*/
void PredictionProcessor::computeRotationMatrixFromNormal(const Eigen::Vector3d &normal, rw::math::Rotation3D<double> &R)
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

    for (size_t i = 0; i < 3; i++)
    {
        R(0,i) = u[i];
        R(1,i) = v[i];
        R(2,i) = n[i];
    }
}

/**
 * @brief Compute centers of connected components in the green channel of an image.
 * @param image The image to compute the centers from.
 * @param dest The vector to save the centers to.
*/
void PredictionProcessor::computeCenters(const cv::Mat &image, std::vector<cv::Point2i> &dest, int max_radius)
{
    /* Isolate green channel */ 
    cv::Mat bgr[3];
    cv::split(image, bgr);
    cv::Mat greenOnly = bgr[1];

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(greenOnly, labels, stats, centroids);

    // Sort centroids by size
    std::vector<std::pair<int, cv::Point2i>> sizeCentroids; // (size, centroid)
    for (int i = 0; i < stats.rows; i++)
    {
        int size = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2i centroidInt = cv::Point2i(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        std::cout << "(size, centroid): (" << size << ", " << centroidInt << ")" << std::endl;
        if (size > max_radius)
        {
            std::cout << "skipping" << std::endl;
            continue;
        }
        sizeCentroids.push_back(std::make_pair(size, centroidInt));
    }
    std::sort(sizeCentroids.begin(), sizeCentroids.end(), [](const std::pair<int, cv::Point2i> &a, const std::pair<int, cv::Point2i> &b) {
        return a.first > b.first;
    });

    // Extract centroids from sizeCentroids into a vector
    dest.resize(sizeCentroids.size());
    for (size_t i = 0; i < dest.size(); i++)
    {
        dest[i] = sizeCentroids[i].second;
        // std::cout << "sorted: " << sizeCentroids[i].second << std::endl;
    }
}

void PredictionProcessor::outlierRemoval(PointCloudPtr &pc, double voxel_size, double std_dev, int nn_max)
{
    // double voxel_size = 0.005;
    auto voxel_down_pcd = pc->VoxelDownSample(voxel_size);
    voxel_down_pcd->RemoveStatisticalOutliers(nn_max, std_dev);
    pc = voxel_down_pcd;
}

// ------------------- Private Methods -------------------
/**
 * @brief Convert a cv::Mat to an open3d::geometry::Image.
 * @param cv_image_in The cv::Mat to convert.
 * @param o3d_image_out The open3d::geometry::Image to save the converted image to.
*/
void PredictionProcessor::cv2o3dImage(const cv::Mat &cv_image_in, open3d::geometry::Image &o3d_image_out)
{
// open3d::geometry::Image depth_image;
    const int width = cv_image_in.cols;
    const int height = cv_image_in.rows;
    o3d_image_out.Prepare(width, height, 1, sizeof(uint16_t));
    const uint16_t* data_ptr = reinterpret_cast<const uint16_t*>(cv_image_in.data);
    std::memcpy(o3d_image_out.data_.data(), data_ptr, width * height * sizeof(uint16_t));
}


