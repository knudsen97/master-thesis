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

    PyObject* tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple, 0, pArray);
    return tuple;
}

Inference::Inference(std::string filepath, std::string function_name)
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
    this->filename_ = filename;
    this->function_name_ = function_name; // This is used in the predict function
}

Inference::~Inference()
{
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_Finalize();
}
