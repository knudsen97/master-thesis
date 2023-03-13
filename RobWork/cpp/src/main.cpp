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


// Standard headers
#include <iostream>
#include <string>

#include <open3d/Open3D.h>


// Include OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/rgbd/linemod.hpp>

// Include "inc" headers
#include "../inc/Sensor.hpp"


int main()
{
    // auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0);
    // sphere->ComputeVertexNormals();
    // sphere->PaintUniformColor({0.0, 1.0, 0.0});
    // open3d::visualization::DrawGeometries({sphere});

    // Create green square in opencv and imshow
    // cv::Mat img = cv::imread("../../../data/color-input/000000-0.png", cv::IMREAD_COLOR);

    // cv::Mat img(512, 512, CV_8UC3, cv::Scalar(0, 255, 0));
    // cv::imshow("Image", img);
    // cv::waitKey(0);

    // std::cout << "Hello World!" << std::endl;
    

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

    RobWorkStudioApp app("");
    RWS_START (app)
    {
        RobWorkStudio* const rwstudio = app.getRobWorkStudio();
        rwstudio->postOpenWorkCell(wcFile);
        TimerUtil::sleepMs(5000);

        const SceneViewer::Ptr gldrawer = rwstudio->getView()->getSceneViewer();
        const GLFrameGrabber::Ptr grabber = ownedPtr(new GLFrameGrabber(width, height, fovy));
        const GLFrameGrabber25D::Ptr grabber25d = ownedPtr(new GLFrameGrabber25D(width, height, fovy));
        grabber->init(gldrawer);
        grabber25d->init(gldrawer);

        SimulatedCamera camera = SimulatedCamera("SimulatedCamera", fovy, camFrame, grabber);
        // camera.getSensor()->getSensorModel()->
        // auto camsens = camera.getCameraSensor();
        // std::string test = camsens->getModelInfo();
        // std::cout << "model info: " << test << std::endl;

        SimulatedScanner25D scanner = SimulatedScanner25D("SimulatedScanner25D", depthFrame, grabber25d);
        SimulatedRGBD RealSense(camera, scanner);
        RealSense.initCamera(100);
        RealSense.initScanner25D(100);

        static const double DT = 0.001;
        const Simulator::UpdateInfo info(DT);

        State state = wc->getDefaultState();

        cv::Mat image;
        RealSense.acquireImage(state, info);
        RealSense.getImage(image, ImageType::BGR);   
        

        PointCloudPtr pc;
        RealSense.acquireDepth(state, info);
        RealSense.getDepth(pc);

        const open3d::geometry::KDTreeSearchParamHybrid search_param(0.1, 30);
        pc->EstimateNormals(search_param);
        pc->NormalizeNormals();

        // Draw circle in middle of image
        cv::Point center = cv::Point(320, 240);
        cv::circle(image, center, 5, cv::Scalar(0, 0, 255), -1);

        // Visualize image and point cloud
        cv::imshow("Image", image);
        open3d::visualization::VisualizerWithKeyCallback o3d_vis;
        o3d_vis.CreateVisualizerWindow("PointCloud", width, height);
        o3d_vis.AddGeometry(pc);
        o3d_vis.Run();



        // Close camera, scanner and RobWorkStudio
        RealSense.close();
        app.close();
    }
    RWS_END()

    cv::waitKey(1000);


    return 0;
}