// Standard headers
#include <iostream>
#include <string>
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include <cmath>
#include <boost/filesystem.hpp>


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
#include <rw/geometry/Sphere.hpp>
#include <rw/kinematics/Frame.hpp>
#include <rw/kinematics/FixedFrame.hpp>
#include <rw/kinematics/MovableFrame.hpp>

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

/**
 * @brief Check if a string is in a vector of strings.
 * @param str The string to check.
 * @param strVector The vector of strings to check against.
 * @return True if the string is in the vector, false otherwise.
*/
bool stringInVector(std::string str, std::vector<std::string> strVector)
{
    for (auto& s : strVector)
    {
        if (str.find(s) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Evaluate the prediction based on blob.
 * @param blobPredictionCenters The predicted centers of the blobs.
 * @param wc The workcell.
 * @return The result of the evaluation. Will be filled with the number of true positives, false positives and false negative in that order.
*/
std::vector<int> evaluateBlobCount(std::vector<rw::math::Vector3D<double>> blobPredictionCenters, const WorkCell::Ptr wc)
{
    std::vector<int> result(3);
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    std::string filePath = wc->getFilePath();

    //get what object is in the scene
    std::string objPath = filePath.substr(0, filePath.find_last_of("/")) + "/objects";
    std::vector<std::string> bodyNames;
    std::string frameName, objectName, bodyName;
    for (auto& p: boost::filesystem::directory_iterator(objPath))
    {
        objectName = p.path().filename().string();
        frameName = objectName;
        frameName[0] = std::toupper(frameName[0]);
        frameName = frameName + "Reference";
        // find object in scene
        if (wc->findFrame(frameName) != nullptr)
        {
            std::cout << "found object: " << objectName << std::endl;
            bodyNames.push_back(objectName + ".Base"); //add .Base to the name to match the frame name in workcell
        }
    }
    rw::kinematics::MovableFrame::Ptr predPointMovableFrame = wc->findFrame<rw::kinematics::MovableFrame>("PredPoint");
    if (predPointMovableFrame == nullptr)
    {
        std::cerr << "PredPoint frame not found" << std::endl;
        return result;
    }
    
    // Create collision detector
    auto collisionStrategy = rwlibs::proximitystrategies::ProximityStrategyYaobi::make();
    auto collisionDetector = rw::common::ownedPtr(new rw::proximity::CollisionDetector(wc, collisionStrategy));
    rw::proximity::CollisionDetector::QueryResult* data = new rw::proximity::CollisionDetector::QueryResult;

    // check if there are collision in default state
    rw::kinematics::State stateCopy = wc->getDefaultState().clone(); // no need to reset since we only move 1 object
    if (collisionDetector->inCollision(stateCopy))
    {
        std::cerr << "Collision in default state" << std::endl;
        return result;
    }

    // move prediction point and find colliding objects
    rw::math::RPY<double> rpy(0, 0, 0);
    std::set<std::string> collisionSet;
    for (auto blobPredictionCenter : blobPredictionCenters)
    {
        // move predPointFrame to predicted location
        rw::math::Transform3D<double> predPointTransform = rw::math::Transform3D<double>(blobPredictionCenter, rpy.toRotation3D());
        predPointMovableFrame->moveTo(predPointTransform, stateCopy);

        // check for collision
        collisionDetector->inCollision(stateCopy, data, false);
        auto collidingFrames = data->getFramePairVector();
        // std::cout << "Looking for collision at point: " << blobPredictionCenter << std::endl;
        // We only expect one collision so we have breaks in the loop
        for (auto& framePair : collidingFrames)
        {
            std::cout << "colliding framePair: " << framePair.first->getName() << " " << framePair.second->getName() << "\tat point: " << blobPredictionCenter << std::endl;
            // get object name from pair
            if(stringInVector(framePair.first->getName(), bodyNames))
            {
                bodyName = framePair.first->getName();
            }
            else if(stringInVector(framePair.second->getName(), bodyNames))
            {
                bodyName = framePair.second->getName();
            }
            else
            {
                falsePositives++;
                break;
            }
            // check if object name exists in collision set
            if (collisionSet.find(bodyName) == collisionSet.end())
            {
                collisionSet.insert(bodyName);
                truePositives++;
            }
            else
            {
                falsePositives++;
            }
            break;
        }
    }
    // check if predictino missed any objects
    for (auto name : bodyNames)
    {
        if (collisionSet.find(name) == collisionSet.end())
        {
            falseNegatives++;
        }
    }

    delete data;
    result[0] = truePositives;
    result[1] = falsePositives;
    result[2] = falseNegatives;
    return result;
}

int main(int argc, char** argv)
{
    std::string model_name = "unet_resnet101_10_l2_e-5_scse_synthetic_data_4000_jit.pt";
    std::string file_name;
    std::string folder_name;

    for (int i = 0; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model_name")
        {
            model_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--file_name")
        {
            file_name = argv[i + 1];
        }
        else if (std::string(argv[i]) == "--folder_name")
        {
            folder_name = argv[i + 1];
        }
    }
    if (file_name.empty())
    {
        file_name = model_name.substr(0, model_name.size() - 7);
    }
    if (folder_name.empty())
    {
        folder_name = file_name;
    }
        
    std::cout << "using model: " << model_name << std::endl;
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
        RealSense.getImage(image, ImageType::BGR);  
        // Inference::change_image_color(image, cv::Vec3b({255, 255, 255}), cv::Vec3b({40,90,120}));
        Inference inf("../../../jit_models/" + model_name);
        auto time_start = std::chrono::high_resolution_clock::now();
        inference_sucess = inf.predict(image, returned_image);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
 
        if (inference_sucess)
        {
            std::cout << "Inference success" << std::endl;
            // cv::imshow("Returned image", returnedImage);
            // cv::waitKey(0);
        }
        else
        {
            std::cerr << "Inference failure" << std::endl;
            RealSense.close();
            app.close();
            return -1;
        }
        
        // Get depth image and point cloud
        PointCloudPtr pc; // This is actually not really used for anything atm. Dont think it is needed?
        cv::Mat depth;
        RealSense.acquireDepth(state, info);
        RealSense.getPointCloudAndDepthImage(pc, depth);

        // Add random noise to depth image 
        // RealSense.addDepthNoise(depth, 0.0, 0.1, 100);

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
        std::cout << "---- finding centroids ----" << std::endl;
        cv::Point center = cv::Point(400, 200);
        std::vector<cv::Point> centers;
        processor.computeCenters(returned_image, centers, 10000);
        bool estimateNormal = !centers.empty();

        Eigen::Vector3d normal;
        Eigen::Vector3d point_3d;
        cv::Point3d center_3d;
        std::vector<cv::Point3d> centers_3d;
        std::vector<rw::math::Vector3D<double>> centers_3d_rw;
        rw::math::Transform3D<double> T_obj_cam;
        std::vector<rw::math::Transform3D<double>> T_obj_cam_vec;
        std::vector<int> tp_fp;
        if (estimateNormal)
        {
            std::cout << "---- Estimate normals ----" << std::endl;
            int centerIndex = 0;
            center = centers[centerIndex];
            // center = centers[centers.size()-1];
            for (auto c : centers)
                std::cout << "center: " << c << std::endl;
            cv::circle(image, center, 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(returned_image, center, 5, cv::Scalar(0, 0, 0), -1);

            // Estimate normals for point cloud and normalize them
            processor.estimateAllNormals(pc_new, 0.05, 30, true);

            // Convert pixel to 3d point
            for (auto c : centers)
            {
                processor.pixel2cam(depth, c, center_3d);
                centers_3d.push_back(center_3d);
                centers_3d_rw.push_back(rw::math::Vector3D<double>(center_3d.x, center_3d.y, center_3d.z));
            }
            processor.pixel2cam(depth, center, center_3d);
            center_3d = centers_3d[centerIndex];
            rw::math::Vector3D<double> center_rw = centers_3d_rw[centerIndex];

            // Find index of closest point in point cloud to 3d center point
            int min_index = processor.findIndexOfClosestPoint(pc_new, center_3d, flip);

            // Get normal and 3d point from closest point in point cloud 
            point_3d = pc_new->points_[min_index];
            normal = pc_new->normals_[min_index];


            // Flip normal if it points away from camera
            if (normal(2) < 0)
                normal = -normal;

            // Manual flipping
            for (auto& c : centers_3d_rw)
            {
                c(0) = -c(0);
                c(1) = c(1);
                c(2) = -c(2);
            }
            center_rw(0) = -center_rw(0);
            center_rw(1) = center_rw(1);
            center_rw(2) = -center_rw(2);

            // Calculating transformation            
            rw::math::Rotation3D<double> R_obj_cam;
            processor.computeRotationMatrixFromNormal(normal, R_obj_cam);
            T_obj_cam = rw::math::Transform3D<double>(center_rw, R_obj_cam);
        }
        else
        {
            std::cout << "No blob found" << std::endl;
        }

        // tp_fp = evaluateBlobCount(centers_3d, wc);

        // ------------------------------------------------------
        // ------------- Visualization --------------------------
        // ------------------------------------------------------
        // Create normal vector line
        std::shared_ptr<open3d::geometry::LineSet> line_ptr;
        if (estimateNormal)
        {
            double scale = 0.1;
            auto line = open3d::geometry::LineSet();
            line.points_.push_back(point_3d);
            line.points_.push_back(point_3d + normal*scale);
            line.lines_.push_back(Eigen::Vector2i(0, 1));
            line.colors_.push_back(Eigen::Vector3d(1, 0, 0));
            line_ptr = std::make_shared<open3d::geometry::LineSet>(line);
        }

        // Create image file names to save files
        std::string image_file_name = file_name + "_image.png";
        std::string returned_image_file_name = file_name + "_returned_image.png";
        std::string point_cloud_file_name = file_name + "_point_cloud.png";
        folder_name = "../images/" + folder_name;
        boost::filesystem::create_directories(folder_name);

        // Visualize image and point cloud
        // cv::imshow("Image", image);
        // cv::imshow("Depth", depth);
        // cv::imshow("Inference", returned_image);
        cv::imwrite(folder_name + '/' + image_file_name, image);
        cv::imwrite(folder_name + '/' + returned_image_file_name, returned_image);
        open3d::visualization::VisualizerWithKeyCallback o3d_vis;
        o3d_vis.CreateVisualizerWindow("PointCloud", width, height);
        o3d_vis.AddGeometry(pc_new);
        if (estimateNormal)
            o3d_vis.AddGeometry(line_ptr);
        o3d_vis.CaptureScreenImage(folder_name + '/' + point_cloud_file_name);
        // o3d_vis.Run();
        // o3d_vis.DestroyVisualizerWindow();

        double valid_point_th = 0.001;
        if(abs(point_3d(0) - center_3d.x) < valid_point_th && abs(point_3d(1) - center_3d.y) < valid_point_th)
        {
            std::cerr << "Invalid point found by inference" << std::endl;
            RealSense.close();
            app.close();
            return -1;
        }

        // transform from world to object
        rw::math::Transform3D<> frameObjTCam = T_obj_cam;
        
        // lets pretend that obj->cam is actually cam->obj (if this does not work, use the above code)
        rw::math::Transform3D<> frameCamTObj = frameObjTCam;

        // Calculate world to object transformation
        rw::math::Transform3D<> frameWorldTCam = camTransform;
        rw::math::Transform3D<> frameWorldTObj = frameWorldTCam * frameCamTObj;

        for (auto& c : centers_3d_rw)
        {
            c = frameWorldTCam * c;
        }
        std::vector<int> evalResults;
        std::cout << "---- Evaluating blob count ----" << std::endl;
        evalResults = evaluateBlobCount(centers_3d_rw, wc);
        std::cout << "True positives: " << evalResults[0] << std::endl;
        std::cout << "False positives: " << evalResults[1] << std::endl;
        std::cout << "False negatives: " << evalResults[2] << std::endl;

        std::cout << "---- inverse kinematics ----" << std::endl;
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

        //finding the solution with the shortest distance to the start
        rw::math::Q Qstart = UR5->getQ(state);
        rw::math::QMetric::Ptr metric = rw::math::MetricFactory::makeEuclidean<rw::math::Q>();
        rw::math::Q Qgoal = collisionFreeSolution[0];
        double distance = metric->distance(Qstart, Qgoal);
        double calculatedDistance = 0;
        for (auto q : collisionFreeSolution)
        {
            calculatedDistance = metric->distance(Qstart, q);
            if (calculatedDistance < distance)
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
        
        // Get path from Qstart to Qgoal
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
// End of main