#include <rw/core/PropertyMap.hpp>
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

#include <iostream>
#include <string>


int main()
{
    std::string wcFile = "../../Project_WorkCell/Scene.wc.xml";

    const WorkCell::Ptr wc = WorkCellLoader::Factory::load(wcFile);
    if (wc.isNull()) 
        RW_THROW ("WorkCell could not be loaded.");
    
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
    std::cout << "Camera properties: fov " << fovy << " width " << width << " height " << height
              << std::endl;

    RobWorkStudioApp app("");
    RWS_START (app)
    {
        RobWorkStudio* const rwstudio = app.getRobWorkStudio();
        rwstudio->postOpenWorkCell(wcFile);
        TimerUtil::sleepMs(5000);

        const SceneViewer::Ptr gldrawer = rwstudio->getView()->getSceneViewer();
        const GLFrameGrabber25D::Ptr grabber = ownedPtr(new GLFrameGrabber25D(width, height, fovy));
        grabber->init(gldrawer);

        SimulatedScanner25D::Ptr scanner = ownedPtr(new SimulatedScanner25D("SimulatedScanner25D", depthFrame, grabber));
        scanner->setFrameRate(100);
        scanner->open();
        scanner->acquire();

        static const double DT = 0.001;
        const Simulator::UpdateInfo info(DT);

        State state = wc->getDefaultState();
        int cnt = 0;

        rw::geometry::PointCloud pc;
        while( !scanner->isScanReady() )
        {
            std::cout << "Waiting for scan to be ready..." << std::endl;
            scanner->update(info, state);
            cnt++;
        }
        std::cout << "Scan ready after " << cnt << " iterations." << std::endl;

        pc = scanner->getScan();

        rw::geometry::PointCloud::savePCD(pc, "mypcd.pcd");
        // rw::geometry::savePCD(pc, "mypcd.pcd");
        // savePCD(pc, "mypcd.pcd")

        scanner->close();
        app.close();
    }
    RWS_END()

    return 0;
}