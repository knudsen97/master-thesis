// Standard C++ includes
#include <iostream>

// Include OpenCV header file
#include <opencv2/opencv.hpp>

// Include Open3D header file
#include <open3d/Open3D.h>

// RobWork includes
#include <rw/math.hpp>
#include <rw/loaders.hpp>
#include <rw/models.hpp>

// Include rtde
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/rtde_control_interface.h>

// Include header files
#include "../inc/Sensor.hpp"
#include "../inc/PredictionProcessor.hpp"
#include "../inc/Inference.hpp"
#include "../inc/CameraCalibration.hpp"
#include "../inc/InverseKinematics.hpp"
#include "../inc/MotionPlanning.hpp"


typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;


void load_intrinsics(const std::string &filename, cv::Mat &intrinsics_out, cv::Mat &dist_coeff_out);
void load_extrinsics(const std::string &filename, cv::Mat &extrinsics);

int main(int argc, char* argv[])
{
    // Check if the camera is connected
    if (!open3d::t::io::RealSenseSensor::ListDevices())
    {
        std::cerr << "No camera detected" << std::endl;
        return -1;
    }

    // Load workcell and device
    std::string wcFile = "../Project_WorkCell/Scene.wc.xml";
    const rw::models::WorkCell::Ptr wc = rw::loaders::WorkCellLoader::Factory::load(wcFile);
    if (wc.isNull()) 
        RW_THROW ("WorkCell could not be loaded.");
    const rw::models::SerialDevice::Ptr UR5 = wc->findDevice<rw::models::SerialDevice>("UR-6-85-5-A");
    if (UR5 == nullptr)
        RW_THROW ("UR5 not found.");
    rw::kinematics::Frame* camFrame = wc->findFrame("Camera_Left");

    // Connecting to UR
    std::string ip = "177.22.22.2";
    ur_rtde::RTDEControlInterface UR5_control(ip);
    ur_rtde::RTDEReceiveInterface UR5_receive(ip);

    // Get and print Qs
    std::vector<double> Qs = UR5_receive.getActualQ();
    std::cout << "Qs: ";
    for (size_t i = 0; i < Qs.size(); i++)
        std::cout << Qs[i] << " ";
    std::cout << std::endl;

    std::vector<double> Q_picture = {-1.15598, -1.71253, -1.58102, -4.56101, -1.56919, -0.76957};

    //-------------------------
    std::string model_file_name = "unet_resnet101_1_jit.pt";
    std::string model_name;
    bool find_extrinsics = false;

    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model")
        {
            model_file_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--model_name")
        {
            model_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--find_extrinsics")
        {
            std::string arg = argv[i + 1];
            // Change string to lowercase.
            std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c){ return std::tolower(c); });
            std::cout << "arg: " << arg << std::endl;
            if (arg == "true")
                find_extrinsics = true;
        }
    }
    if (model_name.empty())
    {
        model_name = model_file_name.substr(0, model_file_name.size() - 7);
    }

    // Create PredictionProcessor object
    double depth_scale = 1e4;
    PredictionProcessor processor(depth_scale);

    // Create Inference object
    cv::Mat image, pred_image, depth;
    cv::Mat returned_image;
    bool inference_sucess;
    Inference inf("../../jit_models/" + model_file_name);

    // Load camera configuration into a Sensor object
    const std::string config_filename = "../config/camera_config.json";
    Sensor sensor(config_filename);
    
    // Start capture
    sensor.startCapture();

    // Create intrinsics and extrinsics matrices
    Eigen::Matrix4d extrinsics_eigen = Eigen::Matrix4d::Identity();
    open3d::camera::PinholeCameraIntrinsic camera_intrinsics;
    cv::Mat intrinsics, extrinsics, dist_coeff;

    // CAMERA CALIBRATION (EXTRINSICS)
    cv::Size board_size(13, 9);
    cv::Size image_size(640, 480);
    int square_size = 19;
    CameraCalibration calib(board_size, image_size, square_size);

    // Load intrinsics and extrinsics from YAML files
    cv::FileStorage fs;
    load_intrinsics("../config/calibration.json", intrinsics, dist_coeff);
    if(!intrinsics.empty())
        sensor.setIntrinsics(intrinsics);
    else // use default intrinsics
        sensor.setIntrinsics(50);

    // Estimate extrinsics of the camera or load from YAML file
    if (find_extrinsics)
        extrinsics = calib.calculateExtrinsics(sensor, "../config/calibration.yaml");
    else // Load extrinsics from YAML file
        load_extrinsics("../config/extrinsics.yaml", extrinsics);

    // sensor.setExtrinsics(extrinsics); // This does not work if the pointcloud
    sensor.setExtrinsics(extrinsics_eigen); // Has to be identity matrix for pointcloud
    sensor.createPinholeCameraIntrinsics(camera_intrinsics);

    // Close all cv windows
    cv::destroyAllWindows();

    // Print intrinsic and extrinsics found/loaded
    std::cout << "Intrinsics:\n" << intrinsics << std::endl;
    std::cout << "Extrinsics:\n" << extrinsics << std::endl;

    // Create flip matrix to flip the point cloud and setup the processor
    Eigen::Matrix4d flip_mat;
    flip_mat << 1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, 1;
    processor.setIntrinsicsAndExtrinsics(camera_intrinsics, extrinsics_eigen);
    processor.setFlipMatrix(flip_mat);

    // kinematics and motion planning variables
    rw::math::Transform3D<double> T_obj_cam;//_rw;
    rw::trajectory::QPath path;

    while(true)
    {
        auto key = cv::waitKey(1);

        // Grab frame
        open3d::t::geometry::Image open3d_image, open3d_depth;
        sensor.grabFrame(open3d_image, open3d_depth);

        // Convert Open3D image to OpenCV image
        sensor.open3d_to_cv(open3d_image, image, true);
        sensor.open3d_to_cv(open3d_depth, depth, false);

        // Just identity at the moment
        sensor.getExtrinsics(extrinsics_eigen);

        // Create point cloud
        auto open3d_depth_legacy = open3d_depth.ToLegacy();
        PointCloudPtr pc;
        pc = open3d::geometry::PointCloud::CreateFromDepthImage(open3d_depth_legacy, camera_intrinsics, extrinsics_eigen, depth_scale);
        pc->Transform(flip_mat);
        

        // Do prediction
        if (key == 'p')
        {
            // Down sample with voxel grid
            processor.outlierRemoval(pc, 0.0005, 1.0, 5);

            // Do inference
            image.copyTo(pred_image);
            auto time_start = std::chrono::high_resolution_clock::now();
            inference_sucess = inf.predict(pred_image, returned_image);
            auto time_end = std::chrono::high_resolution_clock::now();
            std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
    
            // Compute centers
            cv::Point center = cv::Point(400, 200);
            std::vector<cv::Point> centers;
            processor.computeCenters(returned_image, centers, 20000);

            // If the inference was successful and there is at least one center
            if (!inference_sucess && centers.size() <= 0)
            {
                std::cout << "Inference failure" << std::endl;
                continue;
            }
            std::cout << "Inference success" << std::endl;

            // Centers are sorted by size, so the largest center is the one we want
            center = centers[0];
            for (auto c : centers)
                std::cout << "center: " << c << std::endl;
            cv::circle(pred_image, center, 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(returned_image, center, 5, cv::Scalar(0, 0, 0), -1);


            /* ----- Process the prediction ----- */
            // Estimate normals for point cloud and normalize them
            processor.estimateAllNormals(pc, 0.05, 30, true);

            // Convert pixel to 3d point
            cv::Point3d center_3d;
            processor.pixel2cam(depth, center, center_3d);
            std::cout << "center_3d: " << center_3d << std::endl;

            // Find index of closest point in point cloud to 3d center point
            int min_index = processor.findIndexOfClosestPoint(pc, center_3d, true);
            std::cout << "min_index: " << min_index << std::endl;

            // Get normal and 3d point from closest point in point cloud 
            auto point_3d = pc->points_[min_index];
            auto normal = pc->normals_[min_index];
            std::cout << "point_3d: " << point_3d << std::endl;
            std::cout << "normal: " << normal << std::endl;


            // Flip normal if it points away from camera
            if (normal(2) < 0)
                normal = -normal;
            std::cout << "normal: " << normal << std::endl;

            // Compute rotation matrix from normal
            rw::math::Rotation3D<double> R_obj_cam;
            processor.computeRotationMatrixFromNormal(normal, R_obj_cam);

            // manual flipping for test
            // center_3d.x = -center_3d.x;
            center_3d.y = -center_3d.y;
            center_3d.z = -center_3d.z;

            // Create transformation matrix of object in camera frame
            rw::math::Vector3D<double> center_3d_rw(center_3d.x, center_3d.y, center_3d.z);
            T_obj_cam = rw::math::Transform3D<double>(center_3d_rw, R_obj_cam);
            std::cout << "T_obj_cam: \n" << T_obj_cam << std::endl;

            // ------------------------------------------------------
            // ------------- Visualization --------------------------
            // ------------------------------------------------------

            cv::imshow("Image", pred_image);
            cv::imshow("Prediction", returned_image);
            cv::waitKey(0);

            // Create normal vector line
            double scale = 0.01;
            auto line = open3d::geometry::LineSet();
            line.points_.push_back(point_3d);
            line.points_.push_back(point_3d + normal*scale);
            line.lines_.push_back(Eigen::Vector2i(0, 1));
            line.colors_.push_back(Eigen::Vector3d(1, 0, 0));
            auto line_ptr = std::make_shared<open3d::geometry::LineSet>(line);

            open3d::visualization::VisualizerWithKeyCallback o3d_vis;
            o3d_vis.CreateVisualizerWindow("PointCloud", image_size.width, image_size.height);
            o3d_vis.AddGeometry(pc);
            o3d_vis.AddGeometry(line_ptr);
            // o3d_vis.CaptureScreenImage(point_cloud_file_name);
            o3d_vis.Run();

            double valid_point_th = 0.001;
            auto debug_x = abs(point_3d(0) - center_3d.x);
            auto debug_y = abs(point_3d(1) - center_3d.y);
            if( debug_x > valid_point_th || debug_y > valid_point_th)
            {
                std::cerr << "Invalid point found by inference" << std::endl;
                break;
            }
        }

        // Visualization
        cv::imshow("depth", depth);
        cv::imshow("color", image);

        // If you want to do the inverse kinematics and pathplanning, press 'k'
        if (key == 'k')
        {
            if (!UR5_receive.isConnected())
            {
                std::cerr << "UR5_receive not connected" << std::endl;
                break;
            }

            // Inverse kinematics
            rw::kinematics::State state = wc->getDefaultState();
            rw::math::Q q = UR5_receive.getActualQ();
            UR5->setQ(q, state); // set the current configuration to actual configuration
            Eigen::Matrix4d extrinsics_eigen;
            Eigen::Matrix3d rotation_eigen;
            Eigen::Vector3d translation_eigen;
            sensor.getExtrinsics(extrinsics_eigen);

            rotation_eigen = extrinsics_eigen.block<3, 3>(0, 0);
            translation_eigen = extrinsics_eigen.block<3, 1>(0, 3);
            rw::math::Vector3D<double> translation_rw(translation_eigen);
            rw::math::Rotation3D<double> rotation_rw(rotation_eigen);

            rw::math::Transform3D<double> frameWorldTCam(translation_rw, rotation_rw);
            frameWorldTCam = camFrame->wTf(state);
            rw::math::Transform3D<double> frameCamTObj = T_obj_cam;

            // Calculate world to object transformation
            rw::math::Transform3D<double> frameWorldTObj = frameWorldTCam * frameCamTObj;

            InverseKinematics IKsolver(UR5, wc, state);
            double angle_step = 0.1; // increment in roll angle
            double start = -M_PI;
            double end = M_PI;
            if (!IKsolver.solve(frameWorldTObj, start, end, angle_step))
            {
                std::cerr << "No solution found for inverse kinematics goal" << std::endl;
                break;
            }
            std::vector<rw::math::Q> collisionFreeSolution = IKsolver.getSolutions();

            /* begin motion planning */
            MotionPlanning MPsolver(UR5, wc, state);
            rw::math::Q Qstart = UR5->getQ(state);
            rw::math::Q Qtarget = MPsolver.getShortestSolutionFromQToQVec(Qstart, collisionFreeSolution);
            MPsolver.setStartQ(Qstart);
            MPsolver.setTarget(Qtarget);
            MPsolver.setWorldTobject(frameWorldTObj);
            MPsolver.setSubgoalDistance(0.1);

            path = MPsolver.calculateSubgoals();
            // path = MPsolver.getLinearPath(0.5);

            if (!UR5_control.isConnected())
            {
                std::cerr << "UR5_control not connected" << std::endl;
                break;
            }
            auto replay_state = state.clone();
            rw::trajectory::TimedStatePath replayPath;
            double time = 0;
            for (auto q : path)
            {
                UR5->setQ(q, replay_state);
                replayPath.push_back(rw::trajectory::TimedState(time, replay_state));
                time += 0.1;
            }
            
            rw::loaders::PathLoader::storeTimedStatePath(*wc, replayPath, "../Project_WorkCell/RRTPath.rwplay");
            std::cout << "Path saved to RRTPath.rwplay" << std::endl;
            
        }

        if (key == 'r')
        {
            for (auto q : path)
                UR5_control.moveJ(q.toStdVector()); // This line blocks the code until the robot reaches the target
        }

        // If you want to create point cloud press 'p' or close the program with 'q' or 'esc'
        if (key == 't')
        {
            // Down sample with voxel grid
            processor.outlierRemoval(pc, 0.0005, 1.0, 5);

            // Visualize point cloud
            auto resolution = sensor.getResolution();
            open3d::visualization::VisualizerWithKeyCallback o3d_vis;
            o3d_vis.CreateVisualizerWindow("PointCloud", resolution.first, resolution.second);
            o3d_vis.AddGeometry(pc);
            o3d_vis.Run();
        }
        else if (key == 'q' || key == 27)
        {
            break;
        }

        if (key == 'h')
        {
            // Home position
            UR5_control.moveJ(Q_picture);
        }
    }

    sensor.stopCapture();
    UR5_control.disconnect();
    UR5_receive.disconnect();
    
  
    return 0;
}


void load_intrinsics(const std::string &filename, cv::Mat &intrinsics_out, cv::Mat &dist_coeff_out)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["camera_matrix"] >> intrinsics_out;
        fs["distortion_coefficients"] >> dist_coeff_out;
        fs.release();
    }
    else
    {
        std::cout << "No intrinsics file found. Using default intrinsics or run the calibration script." << std::endl;
    }
}

void load_extrinsics(const std::string &filename, cv::Mat &extrinsics)
{
    cv::FileStorage fs("../config/extrinsics.yaml", cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["extrinsics"] >> extrinsics;
        fs.release();
    }
    else
    {
        std::cout << "No extrinsics file found. Using default extrinsics." << std::endl;
    }
}