from sdurw import *
from sdurws import *
# from sdurwsim import *

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from Sensor import FrameGrabber, ColorEncoding


def main():
    # Load workcell
    wc = sdurw.WorkCellLoaderFactory.load('RobWork/Project_WorkCell/Scene.wc.xml')

    if wc.isNull():
        raise Exception("WorkCell could not be loaded")

    # Get device
    device_name = 'UR5-6-85-5-A'
    device = wc.findDevice(device_name)

    if device is None:
        raise Exception("Device could not be found.")

    camera = wc.findFrame("Camera_Left")
    if camera is None:
        raise Exception("Camera frame could not be found.")
    
    scanner25d = wc.findFrame("Scanner25D")
    if scanner25d is None:
        raise Exception("Scanner25D frame could not be found.")

    # Get camera and depth sensor properties and print
    cam_parameters = camera.getPropertyMap().getString("Camera").split(" ")
    depth_parameters = scanner25d.getPropertyMap().getString("Scanner25D").split(" ")
    c_fovy, c_width, c_height = float(cam_parameters[0]),   int(cam_parameters[1]),   int(cam_parameters[2])
    d_fovy, d_width, d_height = float(depth_parameters[0]), int(depth_parameters[1]), int(depth_parameters[2])
    print("Camera properties: fov " + str(c_fovy) + " width " + str(c_width) + " height " + str(c_height))
    print("Scanner25D properties: fov " + str(d_fovy) + " width " + str(d_width) + " height " + str(d_height))

    # Create RobWorkStudio instance
    rwstudio = sdurws.getRobWorkStudioInstance()
    rwstudio.setWorkCell(wc)
    sdurw.sleep(5)
    gldrawer = rwstudio.getView().getSceneViewer()

    framegrabber = sdurw_simulation.ownedPtr( GLFrameGrabber(c_width, c_height, c_fovy) )
    framegrabber.init(gldrawer)

    framegrabber25d = sdurw_simulation.ownedPtr( GLFrameGrabber25D(d_width, d_height, d_fovy) )
    # framegrabber25d = GLFrameGrabber25D(d_width, d_height, d_fovy)
    # framegrabber25dPtr = GLFrameGrabber25DPtr(framegrabber25d)

    try:
        framegrabber25d.init(gldrawer)
    except:
        print("Failed to initialize framegrabber25d")
    # GLFrameGrabber25DCPtr

    # Create simulated camera object
    simcam = SimulatedCamera("SimulatedCamera", c_fovy, camera, framegrabber.asFrameGrabberPtr())
    # simdepth = SimulatedScanner25D("SimulatedScanner25D", scanner25d, framegrabber25d.asFrameGrabberPtr())
    cam_rgb = FrameGrabber("Camera", sim_sensor=simcam)
    # cam_depth = FrameGrabber("Scanner25D", sim_sensor=simdepth)

    DT = 0.001
    counter = 0
    data = np.zeros((480, 640))
    _, ax = plt.subplots()
    im = ax.imshow(data)
    text = ax.text(0.05, 0.95, f"Counter {counter}", transform=ax.transAxes, ha="center")
    plt.title(f"Image from simulated camera")
    plt.ion()
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    

    exitLoop = False
    while not exitLoop:
        info = UpdateInfo(DT)
        state = wc.getDefaultState()

        cam_rgb.acquire_data(state, info, printing=True)
        img = cam_rgb.get_data(ColorFormat=ColorEncoding.RGB, printing=True)

        im.set_data(img)
        text.set_text(f"Counter {counter}")
        plt.pause(0.05)
        counter += 1

        sdurw.sleep(0.5)
        # use matplotlib to check if the user has pressed the 'q' key and exit if so
        if plt.waitforbuttonpress(0.05):
            exitLoop = True
            continue
            # break

    # Shutdown everything
    plt.ioff()
    plt.clf()
    plt.close()

    cam_rgb.release()
    rwstudio.postExit()
    sdurw.sleep(1)

    print("Done")





if __name__ == '__main__':
    main()