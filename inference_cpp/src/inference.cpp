#include "../inc/inference.h"
Inference::Inference(std::string filepath)
{
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
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(append_string.c_str());

    // load python file
    pModule = PyImport_ImportModule(filename.c_str());
    if (!pModule) {
        std::cerr << "Error loading Python module" << std::endl;
    }

}


Inference::~Inference()
{
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_Finalize();
}
