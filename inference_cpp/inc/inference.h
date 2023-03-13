#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Inference {
private:
    PyObject* mat_to_parray(cv::Mat& image);
    int init_numpy();

public:
    /**
     * \brief Constructor for Inference class.
     * \param filepath Path to python file.
     * \param function_name Name of the inference function.
    */
    Inference(std::string filepath, std::string function_name);
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

        init_numpy(); // numpy needs to be initialized for each file using numpy
        // get function from pModule
        pFunc = PyObject_GetAttrString(pModule, this->function_name_.c_str());
        if (!pFunc || !PyCallable_Check(pFunc)) {
            std::cerr << "Error loading Python function" << std::endl;
            return 0;
        }


        // // convert image to numpy array
        PyObject* pArray = mat_to_parray(image);
        PyObject* pArgs = PyTuple_Pack(1, pArray);

        // call the python function and return as pValue
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        if (PyErr_Occurred()) {
            std::cerr << "Error: Python function threw an error" << std::endl;
            PyErr_PrintEx(0);
            PyErr_Clear(); // this will reset the error indicator so you can run Python code again
        }
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
            int dimensions = PyArray_NDIM(resulted_pArray);
            int rows = shape[0];
            int cols = shape[1];
            
            destination = PyArray_To_CvMat<channels, recieving_type>(resulted_pArray, rows, cols);
            
            Py_DECREF(resulted_pArray);
            Py_DECREF(pArray);
            return 1;
        }
        return 0;
        // }
        // else {
        //     std::cerr << "Error: pValue is not an ndarray" << std::endl;
        //     return cv::Mat();
        // }
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
                    result.at<cv::Vec<recieving_type, channels>>(cv::Point(j, i))[k] = data[i * cols * channels + j * channels + k];
                }
            }
        }
        return result;
    }
    
    std::string function_name_;
    std::string filename_;
    PyObject *pModule;
    PyObject *pFunc;
};


