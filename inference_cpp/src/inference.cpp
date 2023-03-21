// #include "../inc/inference.h"


// int Inference::init_numpy()
// {

//     if(PyArray_API == NULL)
//         import_array(); 
//     return 0;
// }

// PyObject* Inference::mat_to_pNpArray(cv::Mat& image)
// {
//     npy_intp dims[3] = { image.rows, image.cols, image.channels() };
//     PyObject* pArray = PyArray_SimpleNew(3, dims, NPY_UINT8);
//     std::memcpy(PyArray_DATA((PyArrayObject*)pArray), image.data, image.total() * image.elemSize());

//     return pArray;
// }
// PyObject* Inference::mat_to_parray(cv::Mat& image)
// {
//     // npy_intp dims[3] = { image.rows, image.cols, image.channels() };
//     PyObject* pList = PyList_New(image.total() * image.channels());

//     for (size_t i = 0; i < image.total(); i++)
//     {
//         for (int c = 0; c < image.channels(); c++)
//         {
//             PyList_SetItem(pList, i * image.channels() + c, PyLong_FromLong(image.data[i * image.channels() + c]));
//         }
//     }

//     return pList;
// }

// Inference::Inference(std::string filepath, std::string inference_function, std::string load_function)
// {
//     PyGILState_Ensure ();
//     this->main_thread_state = PyThreadState_Get();
//     this->pThreadState = Py_NewInterpreter ();

//     PyThreadState_Swap (main_thread_state);
//     PyGILState_Release (PyGILState_STATE::PyGILState_UNLOCKED);

//     TEST_NP_TO_TENSOR("Start of constructor");

//     START_PYTHON_CODE
//     int check = init_numpy();
//         if (check != 0) {
//         std::cerr << "Error initializing numpy" << std::endl;
//     }
//     END_PYTHON_CODE

//     // find the subpath to file
//     int file_idx = filepath.find_last_of('/');
//     std::string sub_path = filepath.substr(0, file_idx);

//     // find where to insert subpath
//     std::string append_string = "sys.path.append(\"\")";
//     int path_insert_idx = append_string.find_last_of('\"');

//     // insert subpath
//     append_string.insert(path_insert_idx, sub_path);

//     // find filename without extension
//     std::string filename = filepath.substr(file_idx + 1);
//     int ext_idx = filename.find_last_of('.');
//     filename = filename.substr(0, ext_idx);

//     // add subpath to python path    
//     std::cout << "Appending Python path: " << sub_path << std::endl;
//     START_PYTHON_CODE
//     PyRun_SimpleString("import sys");
//     PyRun_SimpleString(append_string.c_str());
//     END_PYTHON_CODE

//     // load python file
//     std::cout << "Loading Python file: " << filename << std::endl;
//     START_PYTHON_CODE
//     this->pInfModule = PyImport_ImportModule(filename.c_str());
//     END_PYTHON_CODE
//     if (!this->pInfModule) {
//         std::cerr << "Error loading Python module" << std::endl;
//     }

//     /*get functions from python file*/
//     // load model function
//     START_PYTHON_CODE
//     this->pLoadFunc = PyObject_GetAttrString(this->pInfModule, load_function.c_str());
//     this->pInfFunc = PyObject_GetAttrString(this->pInfModule, inference_function.c_str());
//     END_PYTHON_CODE

//     if (!this->pLoadFunc || !PyCallable_Check(this->pLoadFunc)) {
//         std::cerr << "Error loading Python load function" << std::endl;
//     }

//     if (!this->pInfFunc || !PyCallable_Check(this->pInfFunc)) {
//         std::cerr << "Error loading Python inference function" << std::endl;
//     }

//     // get model
//     START_PYTHON_CODE
//     this->pModel = PyObject_CallObject(this->pLoadFunc, nullptr);
//     if (PyErr_Occurred()) {
//         std::cerr << "Error: Python function threw an error while loading model" << std::endl;
//         PyErr_PrintEx(0);
//         PyErr_Clear(); // this will reset the error indicator so you can run Python code again
//     }
//     END_PYTHON_CODE

//     TEST_NP_TO_TENSOR("End of constructor");
    
// }

// Inference::~Inference()
// {
//     Py_DECREF(this->pInfModule);
//     Py_DECREF(this->pInfFunc);
//     Py_DECREF(this->pLoadFunc);
//     Py_DECREF(this->pModel);

//     if (this->pThreadState) {
//         PyThreadState* ts = PyThreadState_Swap (this->pThreadState);;
//         Py_EndInterpreter (this->pThreadState);

//         PyThreadState_Swap(ts);
//     }
// }
