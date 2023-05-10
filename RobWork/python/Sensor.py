from sdurw import *
from sdurws import *
from sdurwsim import *

import numpy as np
import cv2 as cv
import time

from enum import Enum
class ColorEncoding(Enum):
    GRAY = 0
    RGB = 1
    RGBA = 2
    BGR = 3 
    BGRA = 4  

class FrameGrabber():
    def __init__(self, scanner_type, sim_sensor) -> None:
        self.scanner_type = scanner_type
        self.sim_sensor = sim_sensor
        self.sim_sensor.setFrameRate(100)

        if self.scanner_type == "Camera":
            self.sim_sensor.initialize()
            self.sim_sensor.start()
        elif self.scanner_type == "Scanner25D":
            self.sim_sensor.open()
            if not self.sim_sensor.isOpen():
                raise Exception("Scanner25D could not be opened.")
        else:
            print("Scanner type not supported. Please use 'Camera' or 'Scanner25D'.")


    def rw2image(self, image):
        image_out = np.zeros((image.getHeight(), image.getWidth(), 3), dtype=np.uint8)
        for row in range(image.getHeight()):
            for col in range(image.getWidth()):
                image_out[row, col, 0] = image.getPixelValuei(col, row, 0)
                image_out[row, col, 1] = image.getPixelValuei(col, row, 1)
                image_out[row, col, 2] = image.getPixelValuei(col, row, 2)
        return cv.flip(image_out, 1)

    def acquire_data(self, state, info, printing=False):
        self.sim_sensor.acquire()

        cnt = 0
        if self.scanner_type == "Camera":
            if printing:
                print(f"Acquiring new image from simulated camera")
            if not self.sim_sensor.isImageReady():
                if printing:
                    print("Image is not ready yet. Iteration: ")
            while not self.sim_sensor.isImageReady():
                if printing:
                    print(str(cnt), end=", ")
                self.sim_sensor.update(info, state)
                cnt += 1
            if printing:
                print("")
        elif self.scanner_type == "Scanner25D":
            if printing:
                print(f"Acquiring new depth image from simulated camera")
            if not self.sim_sensor.isScanReady ():
                if printing:
                    print("Scan is not ready yet. Iteration: ")
            while not self.sim_sensor.isScanReady ():
                if printing:
                    print(str(cnt), end=", ")
                self.sim_sensor.update(info, state)
                cnt += 1
            if printing:
                print("")

    def get_data(self, ColorFormat=ColorEncoding.RGB, printing=False):
        data = None
        if self.scanner_type == "Camera":
            if printing:
                print("Getting image from RobWork")

            # Get and convert RobWork image to numpy array
            time_start = time.time()
            image = self.sim_sensor.getImage()
            data = self.rw2image(image)
            time_end = time.time()
            if printing:
                print(f"Time to get image: {time_end - time_start} [s]\n")

            if ColorFormat == ColorEncoding.RGB:
                pass # It is RGB by default
            elif ColorFormat == ColorEncoding.GRAY:
                data = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
            elif ColorFormat == ColorEncoding.BGR:
                data = cv.cvtColor(data, cv.COLOR_RGB2BGR)
        elif self.scanner_type == "Scanner25D":
            if printing:
                print("Getting depth image from RobWork")

            # Get and convert RobWork image to numpy array
            time_start = time.time()
            data = self.sim_sensor.getScan()
            time_end = time.time()
            if printing:
                print(f"Time to get scan: {time_end - time_start} [s]\n")


        return data
    
    def release(self):
        self.sim_sensor.stop()

