# Physical Implementation
This README will go through the requirements of the physical implementation and how to compile the project

## Requirements
To build the project from cpp, the project requires Torch, Open3D, OpenCV, and librealsense. 

### **Torch**
Torch was installed through conda with [ollewelins](https://github.com/ollewelin/Installing-and-Test-PyTorch-C-API-on-Ubuntu-with-GPU-enabled) guide
### **librealsense**
librealsense is an SDK by intel to use their RealSense cameras in C++. The installation guide can be found [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md).

### **Open3D**
Please donwload librealsense as described above before installing Open3D. <br>
Open3D was compiled from [source](http://www.open3d.org/docs/release/compilation.html) with the following flags
```bash
cmake -DBUILD_EIGEN3=ON -D BUILD_LIBREALSENSE=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python -DBUILD_UNIT_TESTS=ON ..
```

### **OpenCV**
OpenCV was a standard install with `sudo apt install libopencv-dev` or compiled using the the guide on [OpenCV official website](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)


## Compiling the project
