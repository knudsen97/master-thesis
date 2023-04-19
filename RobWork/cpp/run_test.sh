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
# "unet_resnet101_10_adamax_jit.pt" \
# "unet_resnet101_10_l2_e-3_jit.pt" \
# "unet_resnet101_10_l2_e-4_jit.pt" \
# "unet_resnet101_10_l2_e-5_jit.pt" \
# "unet_resnet101_10_synthetic_1000_jit.pt" \
# "unet_resnet101_10_synthetic_2000_jit.pt" \
# "unet_resnet101_10_synthetic_3000_jit.pt" \
# "unet_resnet101_10_synthetic_4000_jit.pt" \
# "unet_resnet101_10_l2_e-5_scse_jit.pt" \
"unet_resnet101_10_l2_e-5_scse_synthetic_data_4000_jit.pt" \
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
# "10_adamax" \
# "10_l2_e-3" \
# "10_l2_e-4" \
# "10_l2_e-5" \
# "10_synthetic_1000" \
# "10_synthetic_2000" \
# "10_synthetic_3000" \
# "10_synthetic_4000" \
# "10_l2_e-5_scse" \
"10_l2_e-5_scse_synthetic_data_4000" \
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
# "10_adamax" \
# "10_l2_e-3" \
# "10_l2_e-4" \
# "10_l2_e-5" \
# "10_synthetic_1000" \
# "10_synthetic_2000" \
# "10_synthetic_3000" \
# "10_synthetic_4000" \
# "10_l2_e-5_scse" \
"10_l2_e-5_scse_synthetic_data_4000" \
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
# "puzzle_scene.xml" \
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