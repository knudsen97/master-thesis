#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <rws/pythonpluginloader/PythonRunner.hpp>


    
#define START_PYTHON_CODE {rws::python::PythonLock swap (this->pThreadState);
//this->gstate=PyGILState_Ensure();
    // this->main_thread_state=PyThreadState_Get();

#define END_PYTHON_CODE }
//PyGILState_Release(this->gstate);
    //PyEval_RestoreThread(this->main_thread_state);
    

#define TEST_NP_TO_TENSOR(id) std::cout << "test: " << id << std::endl;\
    START_PYTHON_CODE\
    PyRun_SimpleString("import numpy as np");\
    PyRun_SimpleString("import torchvision.transforms as tf");\
    PyRun_SimpleString("a = tf.ToTensor()(np.ndarray((128, 160, 3), dtype=np.float64))");\
    PyRun_SimpleString("print(a.shape)");\
    END_PYTHON_CODE


class Inference {
private:
    /**
     * \brief Converts a cv::Mat to a standard python nested list.
     * \param image Image to be converted.
     * \return Python list.
    */
    PyObject* mat_to_parray(cv::Mat& image);

    /**
     * \brief Converts a cv::Mat to a numpy array.
     * \param image Image to be converted.
     * \return Numpy array.
    */
    PyObject* mat_to_pNpArray(cv::Mat& image);
    int init_numpy();

public:
    /**
     * \brief Constructor for Inference class.
     * \param filepath Path to python file.
     * \param inference_function Name of the inference function.
     * \param load_function Name of the load function.
    */
    Inference(std::string filepath, std::string inference_function, std::string load_function);
    ~Inference();

    /**
     * \brief Predicts the affordance map from an image.
     * \tparam channels Number of channels in the recieving image. Default is 1.
     * \tparam recieving_type Type of the recieving image. Default is double.
     * \param image Image to be passed to python function.
     * \param destination Image to store the result in.
     * \return If inference from python was executed. True if successful, false otherwise.
     */
    template <int channels = 1, typename recieving_type = double>
    bool predict(cv::Mat image, cv::Mat& destination)
    {

        // check if image is empty
        if (image.empty()) {
            std::cerr << "Error: image is empty" << std::endl;
            return 0;
        }
        START_PYTHON_CODE
        init_numpy(); // numpy needs to be initialized for each file using numpy
        END_PYTHON_CODE
        // // convert image to numpy array
        PyObject* pValue;
        START_PYTHON_CODE
        PyObject* pArray = mat_to_pNpArray(image);
        PyObject* pArgs = PyTuple_New( 2 ); //= PyTuple_Pack(1, pArray);

        std::cout << "test: " << 10 << std::endl; 
        PyRun_SimpleString("import numpy as np");\
        PyRun_SimpleString("import torchvision.transforms as tf");\
        PyRun_SimpleString("a = tf.ToTensor()(np.ndarray((128, 160, 3), dtype=np.float64))");\
        PyRun_SimpleString("print(a.shape)");\

        PyTuple_SetItem(pArgs, 0, pArray);
        PyTuple_SetItem(pArgs, 1, this->pModel);

        // call the python function and return as pValue
        pValue = PyObject_CallObject(this->pInfFunc, pArgs);
        if (PyErr_Occurred()) {
            std::cerr << "Error: Python function threw an error while calling inference" << std::endl;
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
        Py_DECREF(pArgs);
        Py_DECREF(pArray);
        END_PYTHON_CODE

        if (pValue == nullptr) {
            std::cerr << "Error: pValue is NULL" << std::endl;
            return 0;
        }

        // pValue has to be an ndarray
        // if (PyArray_Check(pValue)) { // this causes segmentation fault
        if (pValue != nullptr) {
            PyArrayObject *resulted_pArray = (PyArrayObject *)pValue;
            // convert back from python to c value
            npy_intp *shape = PyArray_SHAPE(resulted_pArray);
            // int dimensions = PyArray_NDIM(resulted_pArray);
            int rows = shape[0];
            int cols = shape[1];

            destination = PyArray_To_CvMat<channels, recieving_type>(resulted_pArray, rows, cols);

            Py_DECREF(resulted_pArray);
            Py_DECREF(pValue);
            return 1;
        }
        return 0;
    }




private:
    /**
     * \brief Converts cv::Mat to numpy array.
     * \tparam channels Number of channels in the recieving image. Default is 1.
     * \tparam recieving_type Type of the recieving image. Default is double.
     * \param array array to be converted.
     * \param rows Number of rows in the array.
     * \param cols Number of columns in the array.
     * \return cv::Mat.
     */
    template<int channels, typename recieving_type>
    cv::Mat PyArray_To_CvMat(PyArrayObject* array, int rows, int cols)
    {

        cv::Mat result(rows, cols, CV_MAKETYPE(cv::DataType<recieving_type>::depth, channels));
        // copy data from python array to cv::Mat
        recieving_type* data = (recieving_type*)PyArray_DATA(array);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < channels; k++) {
                    auto debug = data[i * cols * channels + j * channels + k];
                    result.at<cv::Vec<recieving_type, channels>>(cv::Point(j, i))[k] = debug;
                }
            }
        }
        return result;
    }

    PyObject *pInfModule;
    PyObject *pTorchVisModule;
    PyObject *pInfFunc;
    PyObject *pToTensorFunc;
    PyObject *pLoadFunc;
    PyObject *pModel;
    PyThreadState* pThreadState;
    PyThreadState* main_thread_state;
    PyGILState_STATE gstate;
};


