# Master-thesis
This is the code used for a master thesis.

## Compiling the project
### Requirements
To build the project from cpp, the project requires Torch, Open3D, OpenCV, and Robworks 

#### Torch
Torch was installed through conda with [ollewelins](https://github.com/ollewelin/Installing-and-Test-PyTorch-C-API-on-Ubuntu-with-GPU-enabled) guide

#### Open3D
Open3D was compiled from [source](http://www.open3d.org/docs/release/compilation.html) with the following flags
```bash
cmake -DBUILD_EIGEN3=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python -DBUILD_UNIT_TESTS=ON ..
```

#### OpenCV
OpenCV was a standard install with `sudo apt install libopencv-dev` or compiled using the the guide on [OpenCV official website](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

#### Robworks
Robworks was a standard install following the guide on [Robworks official website](https://www.robwork.dk/installation/ubuntu/)

### Compiling
The inference is seperated from Robworks because of compatibility issues and needs to be compiled seperately. The compiled binarry file need to be located in the binary folder. This can be done with the following commands
```bash
cd RobWork/cpp/inference_bin_generator/build
cmake ..
make
cp ../bin
```

Now the project can be built with the following commands
```bash
cd RobWork/cpp/build
cmake ..
make
```

To run the project, the following command can be used
```bash
cd RobWork/cpp/build
./main
```


## Blender
There is a bash file that must with the terminal inside the blender folder.
Items can be added or removed by removing them from items collection in `synthetic_data_generator.blend` which must be opened with blender.