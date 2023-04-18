#!/bin/bash

#define models to use
models=( \
# "unet_resnet101_1_jit.pt" \
# "unet_resnet101_10_batchnorm_jit.pt" \
# "unet_resnet101_10_extended_baseline_jit.pt" \
# "unet_resnet101_10_full_test_jit.pt" \
# "unet_resnet101_22_batchnorm_jit.pt" \
# "unet_resnet101_22_extended_baseline_jit.pt" \
# "unet_resnet101_10_scse_jit.pt" \
# "unet_resnet101_22_scse_jit.pt" \
# "unet_resnet101_test_run_jit.pt" \
# "unet_resnet101_22_batchnorm_scse_jit.pt" \
# "unet_resnet101_10_batchnorm_scse_jit.pt" \
"unet_resnet101_10_adamax_jit.pt" \
)

#define model names
file_names=( \
# "1" \
# "10_batchnorm" \
# "10_extended_baseline" \
# "10_full_test" \
# "22_batchnorm" \
# "22_extended_baseline" \
# "10_scse" \
# "22_scse" \
# "test" \
# "22_batchnorm_scse" \
# "10_batchnorm_scse" \
"10_adamax" \
)

folder_names=( \
# "1" \
# "10_batchnorm" \
# "10_extended_baseline" \
# "10_full_test" \
# "22_batchnorm" \
# "22_extended_baseline" \
# "10_scse" \
# "22_scse" \
# "test" \
# "22_batchnorm_scse" \
# "10_batchnorm_scse" \
"10_adamax" \
)

#define workcells
workcells=( \
"all_object_scattered.xml" \
# "all_object_cluttered.xml" \
# "4_object_scattered.xml" \
# "4_object_cluttered.xml" \
# "3_object_scattered.xml" \
# "3_object_cluttered.xml" \
# "2_object_scattered.xml" \
# "2_object_cluttered.xml" \
# "creeper_scene.xml" \
# "cube_scene.xml" \
# "kodimagnyl_scene.xml" \
# "original_scene.xml" \
# "panodil_scene.xml" \
# "zendium_scene.xml" \
)

current_working_directory=$(pwd)
cd build

for j in ${!workcells[@]}
do
    # change test scene
    cp -f ../../Project_WorkCell/test_scenes/${workcells[$j]} ../../Project_WorkCell/Scene.wc.xml 
    for i in ${!models[@]}
    do
        echo "Running model: ${models[$i]}"
        echo "File name: ${file_names[$i]}"
        echo "Running command: ./main --model_name ${models[$i]} --file_name ${file_names[$i]} --folder_name ${folder_names[$i]}"
        ./main --model_name ${models[$i]} --file_name ${file_names[$i]} --folder_name ${folder_names[$i]}
        # trim point cloud image
        convert -trim ../images/${folder_names[$i]}/${file_names[$i]}_point_cloud.png ../images/${folder_names[$i]}/${file_names[$i]}_point_cloud.png
    done
done

cd $current_working_directory