# Optimal Sequence Picking Using Deep Learning
This is the code used for our Master Thesis where we have used Deep Learning to try and learn the network an optimal picking sequence in the classical bin picking problem. Contrary to other methods and approaches we have tried to not use any depth data for our training and only feed the network RGB images.
We have limited our research to be with medical packaging and a suction gripper.\
Below you see our network prediction and affordance map of a picking sequence in a scene with scattered objects.

<p float="left">
  <img src="images-and-videos/prediction.gif" width="300" />
  <img src="images-and-videos/inference.gif" width="300" /> 
</p>

Using the inference above we used an UR5e robot for the grasping as seen below.

![SEQUENCE_PICKING](images-and-videos/grabber_with_no_grab.gif)

Full details of our work can be read in our report found [here](link).\
We also have full simulation support for this in [RobWork](https://www.robwork.dk/). Requirements for compiling the project can be found below.

## Requirements
To run the code with C++, the project requires [PyTorch](https://pytorch.org/), [Open3D](http://www.open3d.org/), [OpenCV](https://opencv.org/), and [RobWork](https://www.robwork.dk/). Installation guides of these can be found below.

### **PyTorch**
Torch was installed through conda with [ollewelins](https://github.com/ollewelin/Installing-and-Test-PyTorch-C-API-on-Ubuntu-with-GPU-enabled) guide.

### **Open3D**
Open3D was compiled from [source](http://www.open3d.org/docs/release/compilation.html) with the following flags:
```bash
cmake -DBUILD_EIGEN3=ON -D BUILD_LIBREALSENSE=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON -DGLIBCXX_USE_CXX11_ABI=ON -DPYTHON_EXECUTABLE=/usr/bin/python -DBUILD_UNIT_TESTS=ON ..
```

### **OpenCV**
OpenCV was a standard install with `sudo apt install libopencv-dev` or compiled using the the guide on [OpenCV official website](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).

### **RobWork**
Robworks was a standard install following the guide on [Robworks official website](https://www.robwork.dk/installation/ubuntu/).

## Compiling and running the code
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


## Synthetic Data Generation 
There is a bash file that must with the terminal inside the blender folder.
Items can be added or removed by removing them from items collection in `synthetic_data_generator.blend` which must be opened with blender.