#!/bin/bash
Help()
{
   # Display Help
   echo "Generates synthetic images. Calling the script without any arguments will generate 2 images."
   echo
   echo "Syntax: scriptTemplate [-n|h|r]"
   echo "options:"
   echo "n     The number of images to generate. It will generate 2*n images."
   echo "h     Print this Help."
   echo "r     Removes images from folder depth, images, and label."
}
images=1
# Read the options
while getopts n:hr flag
do
    case "${flag}" in
        n) images=${OPTARG:-1};;
        h) Help
           exit;;
        r) rm ./depth/*.png
           rm ./image/*.png
           rm ./label/*.png
           exit;;
    esac
done

for i in $(seq 0 $(($images-1)))
do
    blender -b synthetic_data_generator.blend --python /home/claus/Documents/sdu/9sem/master/code/master-thesis/blender/blender_script/blender_render_script.py -f 40,41
    echo "$((($i+1)*2)) images generated"
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

done