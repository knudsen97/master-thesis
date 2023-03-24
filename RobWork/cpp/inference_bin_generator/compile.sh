if [[ ${USER} == "claus" ]]; then
    conda activate masters

    cd /home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/inference_bin_generator/build

    /usr/bin/cmake --no-warn-unused-cli -DCMAKE_PREFIX_PATH=/home/claus/pytorch_cpp/libtorch-cxx11-abi-shared-with-deps-latest/libtorch -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -S/home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/inference_bin_generator -B/home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/inference_bin_generator/build -G Ninja

    /usr/bin/cmake --build /home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/inference_bin_generator/build --config Debug --target all --

    cp /home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/inference_bin_generator/build/inference /home/claus/Documents/sdu/9sem/master/code/master-thesis/RobWork/cpp/build
fi

if [[ ${USER} == "kristian" ]]; then
    # conda activate masters

    cd /home/kristian/Documents/SDU/master-thesis/RobWork/cpp/inference_bin_generator/build

    /usr/bin/cmake --no-warn-unused-cli -DCMAKE_PREFIX_PATH=/home/kristian/pytorch_cpp/libtorch  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -S/home/kristian/Documents/SDU/master-thesis/RobWork/cpp/inference_bin_generator -B/home/kristian/Documents/SDU/master-thesis/RobWork/cpp/inference_bin_generator/build -G Ninja

    /usr/bin/cmake --build /home/kristian/Documents/SDU/master-thesis/RobWork/cpp/inference_bin_generator/build --config Debug --target all --

    cp /home/kristian/Documents/SDU/master-thesis/RobWork/cpp/inference_bin_generator/build/inference /home/kristian/Documents/SDU/master-thesis/RobWork/cpp/build
fi
