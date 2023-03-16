from sdurw import *
from sdurws import *
from sdurwsim import *

import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import time

from Sensor import FrameGrabber






def convert_image_to_cv(image):
    image_out = np.zeros((image.getHeight(), image.getWidth(), 3), dtype=np.uint8)
    for row in range(image.getHeight()):
        for col in range(image.getWidth()):
            image_out[row, col, 0] = image.getPixelValuei(col, row, 0)
            image_out[row, col, 1] = image.getPixelValuei(col, row, 1)
            image_out[row, col, 2] = image.getPixelValuei(col, row, 2)
    return cv.cvtColor(cv.flip(image_out, 1), cv.COLOR_RGB2BGR)

def main():
    global DispThreadHasFinished, MainThreadHasFinished,test 
    # Load workcell
    wc = sdurw.WorkCellLoaderFactory.load('RobWork/Project_WorkCell/Scene.wc.xml')

    if wc.isNull():
        raise Exception("WorkCell could not be loaded")

    # Get device
    device_name = 'UR5-6-85-5-A'
    device = wc.findDevice(device_name)

    if device is None:
        raise Exception("Device could not be found.")

    cam_name = 'Camera_Left'
    depth_name = 'Scanner25D'

    camera = wc.findFrame("Camera_Left")
    if camera is None:
        raise Exception("Camera frame could not be found.")
    
    properties = camera.getPropertyMap()
    parameters = properties.getString("Camera").split(" ")
    fovy = float(parameters[0])
    width = int(parameters[1])
    height = int(parameters[2])

    print("Camera properties: fov " + str(fovy) + " width " + str(width) + " height " + str(height))
    rwstudio = sdurws.getRobWorkStudioInstance()
    rwstudio.setWorkCell(wc)
    sdurw.sleep(5)
    gldrawer = rwstudio.getView().getSceneViewer()
    framegrabber = sdurw.ownedPtr( GLFrameGrabber(width, height, fovy) )
    framegrabber.init(gldrawer)

    simcam = SimulatedCamera("SimulatedCamera", fovy, camera, framegrabber.asFrameGrabberPtr())
    # simcam.setFrameRate(100)
    # simcam.initialize()
    # simcam.start()

    cam_rgb = FrameGrabber(cam_name, properties, "Camera", sim_sensor=simcam)
    # cam_depth = FrameGrabber(depth_name, properties, "Scanner25D")

    DT = 0.001
    # info = UpdateInfo(DT)
    # state = wc.getDefaultState()
    cnt = 0
    counter = 0
    data = np.zeros((480, 640))
    _, ax = plt.subplots()
    im = ax.imshow(data)
    text = ax.text(0.05, 0.95, f"Counter {counter}", transform=ax.transAxes, ha="center")
    plt.title(f"Image from simulated camera")
    plt.ion()

    exitLoop = False
    while not exitLoop:
        info = UpdateInfo(DT)
        state = wc.getDefaultState()

        print(f"Acquiring new image({counter}) from simulated camera")
        simcam.acquire()
        if not simcam.isImageReady():
            print("Image is not ready yet. Iteration: ")
        while not simcam.isImageReady():
            print(str(cnt), end=", ")
            simcam.update(info, state)
            cnt += 1
        cnt=0
        print("\n")

        print("Image is ready. Loading image from RobWork...")
        time_start = time.time()
        image = simcam.getImage()
        img = convert_image_to_cv(image)
        time_end = time.time()
        print(f"Time to load image took: {time_end - time_start} seconds")

        # Convert cv image to matplotlib image and show
        img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        im.set_data(img2)
        text.set_text(f"Counter {counter}")
        plt.pause(0.05)
        counter += 1

        # use matplotlib to check if the user has pressed the 'q' key and exit if so
        if plt.waitforbuttonpress(0.05):
            exitLoop = True
            continue
            # break

        sdurw.sleep(0.5)

    # Shutdown everything
    plt.ioff()
    plt.clf()
    plt.close()

    simcam.stop()
    rwstudio.postExit()
    sdurw.sleep(1)

    print("Done")





if __name__ == '__main__':
    main()