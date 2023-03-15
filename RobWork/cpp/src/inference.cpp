#include "../inc/inference.h"


int Inference::init_numpy()
{
    import_array();
    return 0;
}

PyObject* Inference::mat_to_parray(cv::Mat& image)
{
    npy_intp dims[3] = { image.rows, image.cols, image.channels() };
    PyObject* pArray = PyArray_SimpleNew(3, dims, NPY_UINT8);
    std::memcpy(PyArray_DATA((PyArrayObject*)pArray), image.data, image.total() * image.elemSize());

    // PyObject* tuple = PyTuple_New(1);
    // PyTuple_SetItem(tuple, 0, pArray);
    return pArray;
}

Inference::Inference(std::string filepath, std::string inference_function, std::string load_function)
{
    // initialize python
    Py_Initialize();
    auto check = init_numpy();
    if (check != 0) {
        std::cerr << "Error initializing numpy" << std::endl;
    }

    // find the subpath to file
    int file_idx = filepath.find_last_of('/');
    std::string sub_path = filepath.substr(0, file_idx);

    // find where to insert subpath
    std::string append_string = "sys.path.append(\"\")";
    int path_insert_idx = append_string.find_last_of('\"');

    // insert subpath
    append_string.insert(path_insert_idx, sub_path);

    // find filename without extension
    std::string filename = filepath.substr(file_idx + 1);
    int ext_idx = filename.find_last_of('.');
    filename = filename.substr(0, ext_idx);

    // add subpath to python path    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(append_string.c_str());

    // load python file
    pModule = PyImport_ImportModule(filename.c_str());
    if (!pModule) {
        std::cerr << "Error loading Python module" << std::endl;
    }

    /*get functions from python file*/
    // load model function
    this->pLoadFunc = PyObject_GetAttrString(pModule, load_function.c_str());
    if (!this->pLoadFunc || !PyCallable_Check(this->pLoadFunc)) {
        std::cerr << "Error loading Python load function" << std::endl;
    }

    // load inference function
    this->pInfFunc = PyObject_GetAttrString(pModule, inference_function.c_str());
    if (!this->pInfFunc || !PyCallable_Check(this->pInfFunc)) {
        std::cerr << "Error loading Python inference function" << std::endl;
    }

    // get model
    this->pModel = PyObject_CallObject(this->pLoadFunc, nullptr);
    if (PyErr_Occurred()) {
        std::cerr << "Error: Python function threw an error while loading model" << std::endl;
        PyErr_PrintEx(0);
        PyErr_Clear(); // this will reset the error indicator so you can run Python code again
    }
}

Inference::~Inference()
{
    Py_DECREF(this->pModel);
    Py_DECREF(this->pLoadFunc);
    Py_DECREF(this->pInfFunc);
    Py_Finalize();
}
