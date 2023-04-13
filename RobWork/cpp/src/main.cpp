// Standard headers
#include <iostream>
#include <string>
#include <thread>         // std::thread
#include <mutex>          // std::mutex


// Include RobWork headers
#include <rw/core/Ptr.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/kinematics/Kinematics.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rw/loaders/path/PathLoader.hpp>
#include <rw/geometry/PointCloud.hpp>

#include <rw/math/RPY.hpp>
#include <rw/pathplanning.hpp>
#include <rw/math/Rotation3D.hpp>
#include <rw/invkin.hpp>
#include <rw/models/SerialDevice.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/proximity/CollisionStrategy.hpp>

#include <rwlibs/simulation/GLFrameGrabber.hpp>
#include <rwlibs/simulation/GLFrameGrabber25D.hpp>
#include <rwlibs/simulation/SimulatedCamera.hpp>
#include <rwlibs/simulation/SimulatedScanner25D.hpp>
#include <rwlibs/proximitystrategies/ProximityStrategyYaobi.hpp>
#include <rwslibs/rwstudioapp/RobWorkStudioApp.hpp>

#include <rw/math/MetricFactory.hpp>
#include <rwlibs/pathplanners/rrt/RRTPlanner.hpp>
#include <rw/pathplanning/PlannerConstraint.hpp>
#include <rw/trajectory.hpp>

using namespace rw::core;
using namespace rw::common;
using rw::graphics::SceneViewer;
using namespace rw::kinematics;
using rw::loaders::WorkCellLoader;
using rw::models::WorkCell;
using rw::sensor::Image;
using namespace rwlibs::simulation;
using namespace rws;

// Include Open3D headers
#include <open3d/Open3D.h>

// Include OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// Include "inc" headers
#include "../inc/Sensor.hpp"
#include "../inc/PredictionProcessor.hpp"
#include "../inc/Inference.hpp"
#include "../inc/InverseKinematics.hpp"

// void live_cam(cv::Mat& image, std::mutex& cam_mtx)
// {
//     int key = 0;
//     while (cv::ord('q') != key)
//     {
//         while (cam_mtx.try_lock())
//         {
//             cv::imshow("Live camera", image);
//             cv::waitKey(1);
//         }
//     }

    

// }
/**
 * @brief Convert a cv::Mat to a rw::math::Transform3D<double>.
 * @param cv_mat The cv::Mat to convert.
 * @param transform The rw::math::Transform3D<double> to convert to.
 * @return True if the conversion was successful, false otherwise.
*/
bool cvMat_2_robworkTransform(cv::Mat& cv_mat, rw::math::Transform3D<double>& transform)
{
    if (cv_mat.empty())
        {
            std::cout << "cv_mat is empty" << std::endl;
            return false;
        }
    if (cv_mat.channels() != 1)
    {
        std::cout << "cv_mat is not a 1-channel image" << std::endl;
        return false;
    }
    if (cv_mat.rows != 4 || cv_mat.cols != 4)
    {
        std::cout << "cv_mat is not a 4x4 matrix" << std::endl;
        return false;
    }

    rw::math::Rotation3D<double> rot = rw::math::RPY<double>(0, 0, 0).toRotation3D();
    rw::math::Vector3D<double> pos = rw::math::Vector3D<double>(0, 0, 0);
    double cv_debug;

    // std::cout << "cv_mat: \n" << cv_mat << std::endl;
    for (int i = 0; i < cv_mat.rows; i++)
    {
        for (int j = 0; j < cv_mat.cols; j++)
        {  
            if (i < 3 && j < 3)
            {
                cv_debug = cv_mat.at<double>(i, j);
                rot(i, j) = cv_debug;
            }
            else if (i < 3 && j == 3)
            {
                cv_debug = cv_mat.at<double>(i, j);
                pos(i) = cv_debug;
            }
        }
    }
    transform = rw::math::Transform3D<double>(pos, rot);
    return true;
}

int main(int argc, char** argv)
{
    std::string model_file_name = "unet_resnet101_1_jit.pt";
    std::string model_name;

    for (size_t i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model")
        {
            model_file_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--model_name")
        {
            model_name = argv[i + 1];
        }
    }
    if (model_name.empty())
    {
        model_name = model_file_name.substr(0, model_file_name.size() - 7);
    }
        

    // Load workcell
    std::string wcFile = "../../Project_WorkCell/Scene.wc.xml";

    const WorkCell::Ptr wc = WorkCellLoader::Factory::load(wcFile);
    if (wc.isNull()) 
        RW_THROW ("WorkCell could not be loaded.");
    
    Frame* const camFrame = wc->findFrame("Camera_Left");
    if (camFrame == nullptr)
        RW_THROW ("Depth frame not found.");

    Frame* const depthFrame = wc->findFrame("Scanner25D");
    if (depthFrame == nullptr)
        RW_THROW ("Depth frame not found.");

    const PropertyMap& properties = depthFrame->getPropertyMap();
    if (!properties.has("Scanner25D"))
        RW_THROW ("Depth frame does not have a Scanner25D.");

    const rw::models::SerialDevice::Ptr UR5 = wc->findDevice<rw::models::SerialDevice>("UR-6-85-5-A");
    if (UR5 == nullptr)
        RW_THROW ("UR5 not found.");
        
    const std::string parameters = properties.get< std::string > ("Scanner25D");
    std::istringstream iss (parameters, std::istringstream::in);
    double fovy;
    int width;
    int height;
    iss >> fovy >> width >> height;
    std::cout << "Camera/depth properties: fov " << fovy << " width " << width << " height " << height
              << std::endl;
    double fovy_pixel = height / 2 / tan (fovy * (2 * M_PI) / 360.0 / 2.0);


    cv::Mat image;
    cv::Mat returned_image;
    std::mutex cam_mtx;

    // TODO: make a live cam thread that can be updated, by updating image with realsense camera
    // std::thread cam_thread(live_cam, std::ref(image), std::ref(cam_mtx));

    bool inference_sucess;

    RobWorkStudioApp app("");
    RWS_START (app)
    {
        // Get RobWorkStudio instance
        RobWorkStudio* const rwstudio = app.getRobWorkStudio();
        rwstudio->postOpenWorkCell(wcFile);
        TimerUtil::sleepMs(1000);

        // Get the scene viewer
        const SceneViewer::Ptr gldrawer = rwstudio->getView()->getSceneViewer();

        // Create frame grabbers for camera and depth sensor
        const GLFrameGrabber::Ptr grabber = ownedPtr(new GLFrameGrabber(width, height, fovy));
        const GLFrameGrabber25D::Ptr grabber25d = ownedPtr(new GLFrameGrabber25D(width, height, fovy));
        grabber->init(gldrawer);
        grabber25d->init(gldrawer);

        // Create SimulatedRGBD RealSense camera using ideal camera intrinsics for simulation
        SimulatedCamera camera = SimulatedCamera("SimulatedCamera", fovy, camFrame, grabber);
        SimulatedScanner25D scanner = SimulatedScanner25D("SimulatedScanner25D", depthFrame, grabber25d);
        cv::Mat intrinsics = (cv::Mat_<double>(3, 3) << fovy_pixel, 0.0,        width/2.0, 
                                                        0.0,        fovy_pixel, height/2.0, 
                                                        0.0,        0.0,        1.0);
        SimulatedRGBD RealSense(camera, scanner, intrinsics);
        RealSense.initCamera(100);
        RealSense.initScanner25D(100);

        // Create state and update info
        static const double DT = 0.001;
        const Simulator::UpdateInfo info(DT);
        State state = wc->getDefaultState();

        // Get camera extrinsics from camera frame
        cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
        auto camTransform = camFrame->getTransform(state);
        
        // Load R and t into cv::Mat
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                R.at<double>(i, j) = camTransform(i, j);
            }
            t.at<double>(i, 0) = camTransform(i, 3);
        }
        

        // Get image data
        RealSense.acquireImage(state, info);
        cam_mtx.lock();
        RealSense.getImage(image, ImageType::BGR);  
        Inference::change_image_color(image, cv::Vec3b({255, 255, 255}), cv::Vec3b({40,90,120}));
        Inference inf("../../../models/" + model_file_name);
        auto time_start = std::chrono::high_resolution_clock::now();
        inference_sucess = inf.predict(image, returned_image);
        auto time_end = std::chrono::high_resolution_clock::now();
        cam_mtx.unlock();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
 
        if (inference_sucess)
        {
            std::cout << "Inference success" << std::endl;
            // cv::imshow("Returned image", returnedImage);
            // cv::waitKey(0);
        }
        else
            std::cout << "Inference failure" << std::endl;
        
        // Get depth image and point cloud
        PointCloudPtr pc; // This is actually not really used for anything atm. Dont think it is needed?
        cv::Mat depth;
        RealSense.acquireDepth(state, info);
        RealSense.getPointCloudAndDepthImage(pc, depth);

        // Create PredictionProcessor object
        double depth_scale = 1e4;
        PredictionProcessor processor(depth_scale);

        // Set intrinsics and extrinsics
        auto camera_intrinsics = open3d::camera::PinholeCameraIntrinsic(width, 
                                    height, 
                                    intrinsics.at<double>(0, 0), 
                                    intrinsics.at<double>(1, 1), 
                                    intrinsics.at<double>(0, 2), 
                                    intrinsics.at<double>(1, 2)
                                    );
        auto camera_extrinsics = Eigen::Matrix4d::Identity();
        processor.setIntrinsicsAndExtrinsics(camera_intrinsics, camera_extrinsics);
        
        // Set flip matrix to flip point cloud to correct orientation
        Eigen::Matrix4d flip_mat;
            flip_mat << 1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, -1, 0,
                        0, 0, 0, 1;

        processor.setFlipMatrix(flip_mat);

        // Create point cloud from depth image
        PointCloudPtr pc_new;
        bool flip = true;
        processor.createPCFromDepth(depth, pc_new, flip);

        // Draw circle in middle of image
        cv::Point center = cv::Point(400, 200);
        std::vector<cv::Point> centers;
        processor.computeCenters(returned_image, centers, 4000);
        center = centers[0];
        for (auto c : centers)
            std::cout << "center: " << c << std::endl;
        cv::circle(image, center, 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(returned_image, center, 5, cv::Scalar(0, 0, 0), -1);

        // Estimate normals for point cloud and normalize them
        processor.estimateAllNormals(pc_new, 0.05, 30, true);

        // Convert pixel to 3d point
        cv::Point3d center_3d;
        processor.pixel2cam(depth, center, center_3d);

        // Find index of closest point in point cloud to 3d center point
        int min_index = processor.findIndexOfClosestPoint(pc_new, center_3d, flip);

        // Get normal and 3d point from closest point in point cloud 
        auto point_3d = pc_new->points_[min_index];
        auto normal = pc_new->normals_[min_index];


        // Flip normal if it points away from camera
        if (normal(2) < 0)
            normal = -normal;

        cv::Mat R_obj_cam;
        processor.computeRotationMatrixFromNormal(normal, R_obj_cam);
        // std::cout << "R_obj_cam: \n" << R_obj_cam << std::endl;
        
        // manual flipping for test
        center_3d.x = -center_3d.x;
        center_3d.z = -center_3d.z;

        // Create transformation matrix of object in camera frame
        cv::Mat T_obj_cam;
        cv::hconcat(R_obj_cam, cv::Mat(center_3d), T_obj_cam);
        cv::vconcat(T_obj_cam, cv::Mat::zeros(1, 4, CV_64F), T_obj_cam);
        T_obj_cam.at<double>(3, 3) = 1;
        std::cout << "T_obj_cam: \n" << T_obj_cam << std::endl;




        // ------------------------------------------------------
        // ------------- Visualization --------------------------
        // ------------------------------------------------------
        // Create normal vector line
        double scale = 0.1;
        auto line = open3d::geometry::LineSet();
        line.points_.push_back(point_3d);
        line.points_.push_back(point_3d + normal*scale);
        line.lines_.push_back(Eigen::Vector2i(0, 1));
        line.colors_.push_back(Eigen::Vector3d(1, 0, 0));
        auto line_ptr = std::make_shared<open3d::geometry::LineSet>(line);

        // Create image file names to save files
        std::string image_file_name = "../images/" + model_name + "_image.png";
        std::string returned_image_file_name = "../images/" + model_name + "_returned_image.png";
        std::string point_cloud_file_name = "../images/" + model_name + "_point_cloud.png";

        // Visualize image and point cloud
        cv::imshow("Image", image);
        cv::imshow("Depth", depth);
        cv::imshow("Inference", returned_image);
        cv::imwrite(image_file_name, image);
        cv::imwrite(returned_image_file_name, returned_image);
        open3d::visualization::VisualizerWithKeyCallback o3d_vis;
        o3d_vis.CreateVisualizerWindow("PointCloud", width, height);
        o3d_vis.AddGeometry(pc_new);
        o3d_vis.AddGeometry(line_ptr);
        o3d_vis.CaptureScreenImage(point_cloud_file_name);
        o3d_vis.Run();

        if(point_3d(0) - center_3d.x < 0.001 && point_3d(1) - center_3d.y < 0.001)
        {
            std::cerr << "Invalid point found by inference" << std::endl;
            RealSense.close();
            app.close();
            return -1;
        }
        // transform from world to object
        rw::math::Transform3D<> frameObjTCam;
        cvMat_2_robworkTransform(T_obj_cam, frameObjTCam);
        
        /* Invert frameObjTCam */
        // rw::math::Transform3D<> identity = rw::math::Transform3D<double>::identity();
        // rw::math::Transform3D<> frameCamTObj;
        // rw::math::Transform3D<double>::invMult(frameObjTCam, identity, frameCamTObj);

        /* NOTE: it seems to do the same as above the the above is probably correct*/
        // auto rotInverted = frameObjTCam.R().inverse();
        // auto transInverted = rotInverted * -frameObjTCam.P();
        // rw::math::Transform3D<> frameCamTObj = rw::math::Transform3D<double>(transInverted, rotInverted);

        // lets pretend that obj->cam is actually cam->obj (if this does not work, use the above code)
        rw::math::Transform3D<> frameCamTObj = frameObjTCam;
        // Calculate world to object transformation
        rw::math::Transform3D<> frameWorldTCam = camTransform;
        rw::math::Transform3D<> frameWorldTObj = frameWorldTCam * frameCamTObj;

        InverseKinematics solver(UR5, wc);
        // double angle_step = 0.1; // increment in roll angle
        // double start = -M_PI;
        // double end = M_PI;
        if (!solver.solve(frameWorldTObj, M_PI/2))
        {
            std::cerr << "No solution found for inverse kinematics" << std::endl;
            RealSense.close();
            app.close();
            return -1;
        }
        std::vector<rw::math::Q> collisionFreeSolution = solver.getSolutions();
        auto collisionFree = solver.getReplay();

        std::cout << "Number of collision free solutions: " << collisionFreeSolution.size() << std::endl;

        // Create path player
        rw::loaders::PathLoader::storeTimedStatePath(*wc, collisionFree, "../../Project_WorkCell/collision_free.rwplay");

        rw::math::Q Qstart = UR5->getQ(state);

        rw::math::QMetric::Ptr metric = rw::math::MetricFactory::makeEuclidean<rw::math::Q>();
        rw::math::Q Qgoal = collisionFreeSolution[0];
        double distance = metric->distance(Qstart, Qgoal);
        double calculatedDistance = 0;
        for (auto q : collisionFreeSolution)
        {
            calculatedDistance = metric->distance(Qstart, q);
            if (calculatedDistance > distance)
            {
                distance = calculatedDistance;
                Qgoal = q;
            }
        }

        // Create collision detector
        rw::proximity::CollisionStrategy::Ptr collisionStrategy = rwlibs::proximitystrategies::ProximityStrategyYaobi::make();
        rw::proximity::CollisionDetector::Ptr collisionDetector = rw::common::ownedPtr(new rw::proximity::CollisionDetector(wc, collisionStrategy));

        // Initialize planner
        rw::pathplanning::QConstraint::Ptr Qconstraint = rw::pathplanning::QConstraint::make(collisionDetector, UR5, wc->getDefaultState());
        rw::pathplanning::PlannerConstraint plannerConstraint = rw::pathplanning::PlannerConstraint::make(collisionDetector, UR5, state);
        rw::pathplanning::QSampler::QSampler::Ptr sampler = rw::pathplanning::QSampler::QSampler::makeConstrained(
            rw::pathplanning::QSampler::QSampler::makeUniform(UR5), plannerConstraint.getQConstraintPtr()
        );
        rw::pathplanning::QEdgeConstraint::Ptr edgeContrain = rw::pathplanning::QEdgeConstraint::make(Qconstraint.get(), metric, 0.1);
        double step_size = 0.1;
        rw::pathplanning::QToQPlanner::Ptr planner = rwlibs::pathplanners::RRTPlanner::makeQToQPlanner(
            plannerConstraint, 
            sampler, 
            metric, 
            step_size, 
            rwlibs::pathplanners::RRTPlanner::RRTConnect
        );
        
        // get path from Qstart to Qgoal
        rw::trajectory::QPath path;
        rw::trajectory::QPath linIntPath;
        bool pathFound = planner->query(Qstart, Qgoal, path);
        if (pathFound) {
            std::cout << "Path found!" << std::endl;
            std::cout << "Path length: " << path.size() << std::endl;
        } else {
            std::cout << "Path not found!" << std::endl;
        }

        double time_step = 0.1;
        double linIntTimeStep = 1;
        for(unsigned int i = 0; i < path.size()-1; i++)
        {
            rw::trajectory::LinearInterpolator<rw::math::Q> LinInt(path[i], path[i+1], linIntTimeStep);
            for(double dt = 0.0; dt < linIntTimeStep; dt += time_step)
            {
                linIntPath.push_back(LinInt.x(dt));
            }
        }

        // Create path player
        auto stateCopy = wc->getDefaultState();
        double time = 0;
        rw::trajectory::TimedStatePath replayPath;
        for (auto q : linIntPath) {
            UR5->setQ(q, stateCopy);
            replayPath.push_back(rw::trajectory::TimedState(time, stateCopy));
            time += time_step;
        }
        rw::loaders::PathLoader::storeTimedStatePath(*wc, replayPath, "../../Project_WorkCell/RRTPath.rwplay");
        std::cout << "replayfile created" << std::endl;
        

        // Close camera, scanner and RobWorkStudio
        RealSense.close();
        app.close();
    }
    RWS_END()

    std::cout << "Done!" << std::endl;


    return 0;
}
