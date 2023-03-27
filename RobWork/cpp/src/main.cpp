// Standard headers
#include <iostream>
#include <string>
#include <thread>         // std::thread
#include <mutex>          // std::mutex


// Include RobWork headers
#include <rw/core/Ptr.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rwlibs/simulation/GLFrameGrabber.hpp>
#include <rwlibs/simulation/GLFrameGrabber25D.hpp>
#include <rwlibs/simulation/SimulatedCamera.hpp>
#include <rwlibs/simulation/SimulatedScanner25D.hpp>
#include <rwslibs/rwstudioapp/RobWorkStudioApp.hpp>
#include <rw/geometry/PointCloud.hpp>

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
#include "../inc/inference.hpp"

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

int main()
{

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


    const std::string parameters = properties.get< std::string > ("Scanner25D");
    std::istringstream iss (parameters, std::istringstream::in);
    double fovy;
    int width;
    int height;
    iss >> fovy >> width >> height;
    std::cout << "Camera/depth properties: fov " << fovy << " width " << width << " height " << height
              << std::endl;



    cv::Mat image;
    cv::Mat returned_image;
    std::mutex cam_mtx;

    // TODO: make a live cam thread that can be updated, by updating image with realsense camera
    // std::thread cam_thread(live_cam, std::ref(image), std::ref(cam_mtx));

    bool a;

    RobWorkStudioApp app("");
    RWS_START (app)
    {
        // Get RobWorkStudio instance
        RobWorkStudio* const rwstudio = app.getRobWorkStudio();
        rwstudio->postOpenWorkCell(wcFile);
        TimerUtil::sleepMs(3000);

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
        cv::Mat intrinsics = (cv::Mat_<double>(3, 3) << 430.0, 0.0,   320.0, 
                                                        0.0,   430.0, 240.0, 
                                                        0.0,   0.0,   1.0);
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
        auto transform = camFrame->getTransform(state);
        
        // Load R and t into cv::Mat
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                R.at<double>(i, j) = transform(i, j);
            }
            t.at<double>(i, 0) = transform(i, 3);
        }
        

        // Get image data
        RealSense.acquireImage(state, info);
        cam_mtx.lock();
        RealSense.getImage(image, ImageType::BGR);  
        Inference::change_image_color(image, cv::Vec3b({255, 255, 255}), cv::Vec3b({40,90,120}));
        Inference inf("../../../models/unet_resnet101_1_jit.pt");
        auto time_start = std::chrono::high_resolution_clock::now();
        a = inf.predict(image, returned_image);
        auto time_end = std::chrono::high_resolution_clock::now();
        cam_mtx.unlock();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
 
        
        // Close camera, scanner and RobWorkStudio
        RealSense.close();
        app.close();
    }
    RWS_END()

        if (a)
        {
            std::cout << "Success" << std::endl;
            cv::imshow("Returned image", returned_image);
            cv::waitKey(0);
        }
        else
            std::cout << "Failure" << std::endl;

        // bool sucess = inf.predict<3, uint8_t>(image, returned_image);
        // PyEval_RestoreThread(pystate); // Restore the thread state
        
        // Get depth image and point cloud
        // PointCloudPtr pc; // This is actually not really used for anything atm. Dont think it is needed?
        // cv::Mat depth;
        // RealSense.acquireDepth(state, info);
        // RealSense.getPointCloudAndDepthImage(pc, depth);

        // // Create PredictionProcessor object
        // double depth_scale = 1e4;
        // PredictionProcessor processor(depth_scale);

        // // Set intrinsics and extrinsics
        // auto camera_intrinsics = open3d::camera::PinholeCameraIntrinsic(width, height, intrinsics.at<double>(0, 0), intrinsics.at<double>(1, 1), intrinsics.at<double>(0, 2), intrinsics.at<double>(1, 2));
        // auto camera_extrinsics = Eigen::Matrix4d::Identity();
        // processor.setIntrinsicsAndExtrinsics(camera_intrinsics, camera_extrinsics);
        
        // // Set flip matrix to flip point cloud to correct orientation
        // Eigen::Matrix4d flip_mat;
        //     flip_mat << 1, 0, 0, 0,
        //                 0, -1, 0, 0,
        //                 0, 0, -1, 0,
        //                 0, 0, 0, 1;

        // processor.setFlipMatrix(flip_mat);

        // // Create point cloud from depth image
        // PointCloudPtr pc_new;
        // bool flip = true;
        // processor.createPCFromDepth(depth, pc_new, flip);

        // // Draw circle in middle of image
        // cv::Point center = cv::Point(400, 200);
        // cv::circle(image, center, 5, cv::Scalar(0, 0, 255), -1);

        // // Estimate normals for point cloud and normalize them
        // processor.estimateAllNormals(pc_new, 0.05, 30, true);

        // // Convert pixel to 3d point
        // cv::Point3d center_3d;
        // processor.pixel2cam(depth, center, center_3d);

        // // Find index of closest point in point cloud to 3d center point
        // int min_index = processor.findIndexOfClosestPoint(pc_new, center_3d, flip);

        // // Get normal and 3d point from closest point in point cloud 
        // auto point_3d = pc_new->points_[min_index];
        // auto normal = pc_new->normals_[min_index];

        // // Flip normal if it points away from camera
        // if (normal(2) < 0)
        //     normal = -normal;

        // cv::Mat R_obj_cam;
        // processor.computeRotationMatrixFromNormal(normal, R_obj_cam);
        // std::cout << "R_obj_cam: \n" << R_obj_cam << std::endl;

        // // Create transformation matrix of object in camera frame
        // cv::Mat T_obj_cam;
        // cv::hconcat(R_obj_cam, cv::Mat(center_3d), T_obj_cam);
        // cv::vconcat(T_obj_cam, cv::Mat::zeros(1, 4, CV_64F), T_obj_cam);
        // T_obj_cam.at<double>(3, 3) = 1;
        // std::cout << "T_obj_cam: \n" << T_obj_cam << std::endl;




        // // ------------------------------------------------------
        // // ------------- Visualization --------------------------
        // // ------------------------------------------------------
        // // Create normal vector line
        // double scale = 0.1;
        // auto line = open3d::geometry::LineSet();
        // line.points_.push_back(point_3d);
        // line.points_.push_back(point_3d + normal*scale);
        // line.lines_.push_back(Eigen::Vector2i(0, 1));
        // line.colors_.push_back(Eigen::Vector3d(1, 0, 0));
        // auto line_ptr = std::make_shared<open3d::geometry::LineSet>(line);

        // // Visualize image and point cloud
        // cv::imshow("Image", image);
        // cv::imshow("Depth", depth);
        // cv::imshow("Inference", returned_image);
        // open3d::visualization::VisualizerWithKeyCallback o3d_vis;
        // o3d_vis.CreateVisualizerWindow("PointCloud", width, height);
        // o3d_vis.AddGeometry(pc_new);
        // o3d_vis.AddGeometry(line_ptr);
        // o3d_vis.Run();





    // std::cout << "Press any key to exit" << std::endl;
    // cv::waitKey(0);
    std::cout << "Done!" << std::endl;


    return 0;
}