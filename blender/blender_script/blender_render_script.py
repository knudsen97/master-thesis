import bpy
import os
import math
import numpy as np
import sys
from mathutils import Vector
from copy import copy
import importlib
import random

try:
    import gt_face_change_script
    importlib.reload(gt_face_change_script)
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

def delete_all_boxes(collection = "copies"):
    """detete all boxes in a collection

    Args:
        collection (str, optional): Name of the collection. Defaults to "copies".
    """
    # Get all objects in the scene
    objects = bpy.data.collections[collection].all_objects
    # Delete all objects that have the name "box" excluding the original box
    for obj in objects:
        bpy.data.objects.remove(obj)

def add_box_at_random(number_of_boxes, _boxes):
    """Generate a random amount of boxes arround the container

    Args:
        number_of_boxes (int): Number of boxes to add
        _boxes (list): List of all boxes that can be added
    """
    original_location = Vector((0.09, 0.5, 0.16))
    box_size = (0.05, 0.09, 0.02) # (x, y, z)
    offset = 0.01
    x_min = original_location[0]
    x_max = original_location[0] + box_size[0]*5+offset
    y_min = original_location[1] - box_size[1]*4-offset
    y_max = original_location[1]
    z_min = original_location[2]
    z_max = original_location[2] + box_size[2]*7+offset

    # Get list of item in collection boxes
    boxes = copy(_boxes)
    random.shuffle(boxes)


    added_boxes = []
    # Add boxes to the scene
        # bpy.context.scene.rigidbody_world.enabled = True


    for i in range(0, number_of_boxes):
        # Shuffle list of boxes

        if len(boxes) == 0:
            boxes = copy(_boxes)
            random.shuffle(boxes)
        original_box = boxes.pop()


        # Generate a random location
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        locations = Vector((x, y, z))

        # Duplicate the original box
        new_box = original_box.copy()
        new_box.data = original_box.data.copy()
        new_box.animation_data_clear()

        # Set the location of the new box
        new_box.location = locations
        new_box.rigid_body.enabled = True

        # Add the new box to the scene in collection 'copies'
        bpy.data.collections['copies'].objects.link(new_box)
        added_boxes.append(new_box)

    return added_boxes


def face_change_handler(FaceColor: gt_face_change_script.FaceColor):
    """Creates the handler that changes the face of the boxes
    """
    def handler(scene):
        """Handler that changes the face of the boxes"""
        frame_num = scene.frame_current
        if frame_num > 40 and not FaceColor.changed:
            FaceColor.paint_gt()    
        elif FaceColor.changed:
            FaceColor.return_face_texture()
    return handler

def main():

    # Reset scene
    # bpy.context.scene.rigidbody_world.enabled = False
    rng = np.random.randint(1, 50, 1).item() # Random number between 0 and 5*5*4
    try:
        delete_all_boxes()
    except NameError:
        pass

    # box = bpy.data.objects['box']
    # boxes = [box]
    boxes = [obj for obj in bpy.data.collections['boxes'].all_objects]
    
    added_boxes = add_box_at_random(rng, boxes)

    face_changer = gt_face_change_script.FaceColor(added_boxes)

    # append face change handler
    def register(handler_function):
        bpy.app.handlers.frame_change_post.append(handler_function)

    register(face_change_handler(face_changer))

    try:
        # Free baking
        bpy.ops.ptcache.free_bake_all()
    except:
        pass


    # Bake animation
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()

    # def update():
    # bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    bpy.context.scene.rigidbody_world.point_cache.frame_end = 40
    bpy.context.scene.rigidbody_world.point_cache.frame_step = 1
    bpy.ops.ptcache.bake_all()


    # Jump to frame 40
    bpy.context.scene.frame_set(40)
    bpy.context.scene.frame_set(41)


if __name__ == "__main__":
    main()