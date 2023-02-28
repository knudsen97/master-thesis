#!/bin/bash

# Make for loop based on arguments
# for i in "$@"
# do
#     blender -b synthetic_data_generator.blend --python /home/claus/Documents/sdu/9sem/master/code/master-thesis/blender/blender_script/blender_render_script.py -f 40,41

#     # Remove nonsens image
#     rm ./depth/*41*.png
#     rm ./label/*41*.png
#     rm ./image/*40*.png

#     # Rename image
#     mv ./depth/*_L.png ./depth/depth$((i*2)).png
#     mv ./depth/*_R.png ./depth/depth$((i*2+1)).png

#     mv ./label/*_L.png ./label/label$((i*2)).png
#     mv ./label/*_R.png ./label/label$((i*2+1)).png

#     mv ./image/*_L.png ./image/image$((i*2)).png
#     mv ./image/*_R.png ./image/image$((i*2+1)).png

#     # mv ./depth/*_L.png ./depth/depth1.png
#     # mv ./depth/*_R.png ./depth/depth2.png

#     # mv ./label/*_L.png ./label/label1.png
#     # mv ./label/*_R.png ./label/label2.png

#     # mv ./image/*_L.png ./image/image1.png
#     # mv ./image/*_R.png ./image/image2.png
# done

# print i from 1 to argument given
for i in $(seq 0 $(($1-1)))
do
    blender -b synthetic_data_generator.blend --python /home/claus/Documents/sdu/9sem/master/code/master-thesis/blender/blender_script/blender_render_script.py -f 40,41
    echo "Image $i generated"
    # Remove nonsens image
    rm ./depth/*41*.png
    rm ./label/*41*.png
    rm ./image/*40*.png

    # Rename image
    mv ./depth/*_L.png ./depth/depth$((i*2)).png
    mv ./depth/*_R.png ./depth/depth$((i*2+1)).png

    mv ./label/*_L.png ./label/label$((i*2)).png
    mv ./label/*_R.png ./label/label$((i*2+1)).png

    mv ./image/*_L.png ./image/image$((i*2)).png
    mv ./image/*_R.png ./image/image$((i*2+1)).png

    # mv ./depth/*_L.png ./depth/depth1.png
    # mv ./depth/*_R.png ./depth/depth2.png

    # mv ./label/*_L.png ./label/label1.png
    # mv ./label/*_R.png ./label/label2.png

    # mv ./image/*_L.png ./image/image1.png
    # mv ./image/*_R.png ./image/image2.png
done