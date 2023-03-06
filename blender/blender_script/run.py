import os
import sys
import importlib

try:
    import gt_face_change_script
    import test_for_one_box
    importlib.reload(gt_face_change_script)
    importlib.reload(test_for_one_box)
except ImportError:
    # Include the path to the blender_script folder
    path_to_file = os.path.dirname(__file__)
    
    # Delete the last part of the path
    path_to_file = path_to_file.split("/")
    path_to_file = path_to_file[:-1]
    path_to_file = "/".join(path_to_file)
    
    # Add blender_script to the path
    path_to_file = path_to_file + "/blender_script"
    
    # Add the path to the sys.path and import
    sys.path.append(path_to_file)
    import gt_face_change_script

print("----------------- new run -----------------")

face_changer = gt_face_change_script.FaceColor()
face_changer.paint_gt()
face_changer.return_face_texture()