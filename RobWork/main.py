# import sdurw, sdurws, sdurwsim
from sdurw import *
from sdurws import *
from sdurwsim import *



import numpy as np
import cv2 as cv

import ctypes
from io import BytesIO
from PIL import Image  
import base64


from enum import Enum

class ColorEncoding(Enum):
    GRAY = 0
    RGB = 1
    RGBA = 2
    BGR = 3 
    BGRA = 4  



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
    sdurw.sleep(5)
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
    # print("image type:" , type(image))
    image.saveAsPPM("RobWork/Project_WorkCell/temp.ppm")
    # img = cv.imread("RobWork/Project_WorkCell/temp.ppm", cv.IMREAD_COLOR)

    camera_frame = wc.findFrame("Camera_Left")
    framegrabber.grab(camera_frame, state)
    image = framegrabber.getImage()

    # sdurw.sleep(1)


    
    image_data = image.getImageData()
    image_data_size = image.getDataSize() #480*640 * (bits_per_channel // 8) *3 #     bits_per_channel = image.getBitsPerPixel()
    print("image_data_size: ", image_data_size)
    print("colorcode: ", ColorEncoding(image.getColorEncoding()).name)
    print("# of channels: ", image.getNrOfChannels())
    sbyte_data = ctypes.string_at(image_data, image_data_size)
    # print("sbyte_data: ", sbyte_data[:10])
    # print("sbyte_data type: ", sbyte_data[:10])
    # sbyte_data.decode('ascii', 'ignore')

    # nparr = np.asarray(bytearray(image_data.read()))
    # print("nparr: ", nparr[:10])
    nparr = np.frombuffer(sbyte_data, np.uint8)
    # img = cv.imdecode(nparr, 1)

    img = nparr.reshape((image.getHeight(), image.getWidth(), 3))
    # cv.imwrite("RobWork/Project_WorkCell/temp2.jpg", img)

    imageData = ctypes.cast(image_data, ctypes.POINTER(ctypes.c_ubyte * image_data_size)).contents
    print("imageData: ", imageData[:10])
    print("imageData type: ", type(imageData))
    imageBytes = bytes(imageData)
    print("imageBytes: ", imageBytes[:10])
    print("imageBytes type: ", type(imageBytes))

    nparr2 = np.frombuffer(imageBytes, np.uint8)
    img2 = nparr2.reshape((image.getHeight(), image.getWidth(), 3))

    cv.imwrite("RobWork/Project_WorkCell/temp2.png", img2)

    cv.imshow("Image", img2)
    cv.waitKey(0)

    # sbyte_data = image_data#.decode()
    # image_data_bytes = BytesIO(sbyte_data.encode('ISO-8859-1', 'ignore'))
    # img = Image.open(image_data_bytes)


    # bytes = sbyte_data.decode('utf-8', 'ignore')
    # bytes = base64.b64decode(sbyte_data) 
    # a numpy array
    # print("bytes type: ", type(bytes))
    # print("bytes length: ", len(bytes))
    # print("bytes: ", bytes[0:10])


    # img = cv.imdecode(np_array, cv.IMREAD_UNCHANGED) 
    # img = np_array.reshape((image.getHeight(), image.getWidth(), 3))

    # Read sbtye_data as a PIL image using frombytes
    # img = Image.frombytes('RGB', (image.getWidth(), image.getHeight()), sbyte_data)
    # img.save("RobWork/Project_WorkCell/temp2.jpg")

    # # Print out some pixel values
    # print("Pixel values:")
    # for i in range(0, 10):
    #     print(img.getpixel((i, i)))


    # img.show()




    # # Check if the image is valid
    # if img is None:
    #     print("Error: Failed to decode image data.")
    # else:
    #     # Display the image
    #     cv.imshow("image", img)
    #     cv.waitKey(0)

    simcam.stop()
    rwstudio.postExit()
    sdurw.sleep(1)

    # cv.imshow("Image", img)
    # cv.waitKey(0)

    # sdurw

if __name__ == '__main__':
    main()