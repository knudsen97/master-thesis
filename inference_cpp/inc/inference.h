#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


class Inference {
private:
    const std::string PYTHON_INFERENCE_FUNCTION_NAME = "predict";
public:
    Inference(std::string filepath);
    ~Inference();

    template<int channels>
    cv::Mat PyArray_To_CvMat(PyArrayObject* array, int rows, int cols)
    {
        cv::Mat result(rows, cols, CV_32FC(channels));
        // copy data from python array to cv::Mat
        float* data = (float*)PyArray_DATA(array);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < channels; k++) {
                    result.at<cv::Vec<float, channels>>(i, j)[k] = data[i * cols * channels + j * channels + k];
                }
            }
        }

        return result;
    }

    template <int channels>
    cv::Mat predict(cv::Mat image)
    {
        // get function from pModule
        pFunc = PyObject_GetAttrString(pModule, PYTHON_INFERENCE_FUNCTION_NAME.c_str());
        if (!pFunc || !PyCallable_Check(pFunc)) {
            std::cerr << "Error loading Python function" << std::endl;
        }

        // convert image to numpy array
        cv::Mat image_float;
        image.convertTo(image_float, CV_32F);
        cv::Mat image_reshaped = image_float.reshape(1, 1);
        npy_intp image_size[2] = { image_reshaped.rows, image_reshaped.cols };
        PyObject* pArray = PyArray_SimpleNewFromData(2, image_size, NPY_FLOAT32, image_reshaped.data);
        PyObject* pArgs = PyTuple_Pack(1, pArray);

        // call the python function and return as pValue
        
        PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
        

        // pValue has to be an ndarray
        if (PyArray_Check(pValue)) {
            PyArrayObject *pArray = (PyArrayObject *)pValue;
            // convert back from python to c value
            npy_intp *shape = PyArray_SHAPE(pArray);
            int dimensions = PyArray_NDIM(pArray);
            int rows = shape[0];
            int cols = shape[1];
            
            cv::Mat result = PyArray_To_CvMat<channels>(pArray, rows, cols);
            
            Py_DECREF(pArray);
            return result;
        }
        else {
            std::cerr << "Error: pValue is not an ndarray" << std::endl;
            return cv::Mat();
        }
    }

private:

    PyObject *pModule;
    PyObject *pFunc;
};


