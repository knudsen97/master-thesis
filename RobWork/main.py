# import sdurw, sdurws, sdurwsim

from sdurw import *
# from sdurw_simulation import *

from sdurws import *
from sdurwsim import *
import sys

import numpy as np

import cv2 as cv

import ctypes

# class Camera():
#     def __init__(self, camera_name):
#         self.camera_name = camera_name
#         self.camera = wc.findFrame(camera_name)




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
    
    properties = camera.getPropertyMap()
    parameters = properties.getString("Camera").split(" ")
    fovy = float(parameters[0])
    width = int(parameters[1])
    height = int(parameters[2])

    print("Camera properties: fov " + str(fovy) + " width " + str(width) + " height " + str(height))
    rwstudio = sdurws.getRobWorkStudioInstance()
    rwstudio.setWorkCell(wc)
    sdurw.sleep(2)
    gldrawer = rwstudio.getView().getSceneViewer()
    framegrabber = sdurw.ownedPtr( GLFrameGrabber(width, height, fovy) )
    framegrabber.init(gldrawer)

    simcam = SimulatedCamera("SimulatedCamera", fovy, camera, framegrabber.asFrameGrabberPtr())
    simcam.setFrameRate(100)
    simcam.initialize()
    simcam.start()
    simcam.acquire()

    DT = 0.001
    info = UpdateInfo(DT)
    state = wc.getDefaultState()
    cnt = 0

    while not simcam.isImageReady():
        print("Image is not ready yet. Iteration: " + str(cnt))
        simcam.update(info, state)
        cnt += 1
    
    image = simcam.getImage()
    print("image type:" , type(image))

    # Convert image to OpenCV format
    image_data = image.getImageData()
    image_encoding = image.getColorEncoding()
    print("image encoding:" , image_encoding)

    frame = wc.findFrame("Camera_Left")
    framegrabber.grab(frame, state)
    rw_image = framegrabber.getImage()

    # Create numpy array with width, height specified a
    img = np.ndarray(shape=(image.getHeight(), image.getWidth(), 3), dtype=np.uint8, buffer=sdurw_core.AnyPtr(rw_image))




    simcam.stop()
    rwstudio.postExit()
    sdurw.sleep(1)

    cv.imshow("Image", img)
    cv.waitKey(0)

    # sdurw

if __name__ == '__main__':
    main()