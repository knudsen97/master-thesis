import bpy
import os
import math
import numpy as np
import sys
from mathutils import Vector

def delete_all_boxes():
    """detete all boxes except the original box
    """
    # Get all objects in the scene
    objects = bpy.data.objects
    # Delete all objects that have the name "box" excluding the original box
    for obj in objects:
        if obj.name.startswith("box") and obj.name != "box":
            bpy.data.objects.remove(obj)

def add_box_at_random(number_of_boxes):
    """Generate a random amount of boxes arround the container

    Args:
        number_of_boxes (int): Number of boxes to add, max 5*5*4
    """
    original_location = Vector((0.09, 0.5, 0.16))
    box_size = (0.05, 0.09, 0.02) # (x, y, z)
    offset = 0.01

    # Get instance of box
    original_box = bpy.data.objects["box"]

    # make array of possible box locations
    x = original_location[0]
    y = original_location[1]
    z = original_location[2]
    x = np.arange(
        original_location[0], 
        original_location[0] + box_size[0]*5+offset, 
        box_size[0]+offset
        ) 
    y = np.arange( # Here we go in the negative direction
        original_location[1] - box_size[1]*4-offset, # y direction is a bit tight
        original_location[1], 
        box_size[1]+offset
        ) 
    z = np.arange(
        original_location[2], 
        original_location[2] + box_size[2]*6+offset, 
        box_size[2]+offset
        )

    # Make a list of all possible locations
    locations = []
    for i in x:
        for j in y:
            for k in z:
                locations.append((i, j, k))
    
    # Shuffle the locations
    np.random.shuffle(locations)

    # Add boxes to the scene
    for i in range(0, number_of_boxes):
        # Duplicate the original box
        new_box = original_box.copy()
        new_box.data = original_box.data.copy()
        new_box.animation_data_clear()

        # Set the location of the new box
        new_box.location = locations[i]

        # Add the new box to the scene
        bpy.context.scene.collection.objects.link(new_box)





def main():
    # Try to import the gt_face_change_script
    try:
        import gt_face_change_script
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

    # Reset scene
    # bpy.context.scene.rigidbody_world.enabled = False
    rng = np.random.randint(0, 5*5*4, 1).item() # Random number between 0 and 5*5*4
    delete_all_boxes()
    add_box_at_random(rng)

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
    # Wait for frame to set
    gt_face_change_script.main()
    bpy.context.scene.frame_set(1)

    # call update in 1 sec
    # bpy.app.timers.register(update, first_interval=1)
    # bpy.context.scene.rigidbody_world.enabled = False
    # # Render the image
    # bpy.ops.render.render(write_still=True)

    # # jump to frame 41
    # bpy.context.scene.frame_set(41)

    # # Render the image
    # bpy.ops.render.render(write_still=True)





    # gt_face_change_script.main()

if __name__ == "__main__":
    main()