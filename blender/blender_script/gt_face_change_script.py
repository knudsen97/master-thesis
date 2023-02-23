import bpy
import os
import math
import numpy as np

OBJECT_ANGLE_SUCTION_THRESHOLD = 1/10 * math.pi

# face_coord_gt_red = np.array([[0.5, 0.00], [0.5, 1.00], [0.00, 1.00], [0.00, 0.00]])
face_coord_gt_red = np.array([[0.00, 0.00], [0.0, 1.00], [0.50, 1.00], [0.50, 0.00]])

face_coord_gt_green = np.array([[0.5, 0.00], [0.5, 1.00], [1.00, 1.00], [1.00, 0.00]])



def classify_faces(faces):
    """Classify faces as green or red (graspable or not graspable)

    Args:
        faces (bpy.object.data.polygon): Only expecting 8 faces

    Returns:
        touple: Two list of faces, green and red
    """
    green_faces = []
    red_faces = []
    # Find the angle of each faces
    for face in faces:
        # Get the normal of the face
        normal = face.normal

        angle = math.acos(normal.dot((0, 0, 1))) # Calculate the angle relative to the XY plane
        # if obj.name == "box.086":
        #     print(f"Box: {obj.name} face: {face.index} angle: {angle} radians")
        if -OBJECT_ANGLE_SUCTION_THRESHOLD < angle < OBJECT_ANGLE_SUCTION_THRESHOLD:
            green_faces.append(face)
        else:
            red_faces.append(face)
    return green_faces, red_faces


def update_face_color(faces, uv_layer, color):
    """Update the color of the faces
    
    Args:
        faces (bpy.object.data.polygon): Only expecting 8 faces
        uv_layer (obj.data.uv_layers.active.data): Uv layer of the object
        color (np.array): Color to set the faces to
        
    Returns:
        None
    """
    vert_indices = 4 # Only expect to be 4 vertices per face
    for face in faces:
        for vert_idx in range(vert_indices):
            uv_layer[face.loop_start + vert_idx].uv = color[vert_idx] # Set all uv coordinates to green

def get_top_faces(mesh):
    """Get the top faces of the mesh
    
    Args:
        mesh (bpy.object.data): Mesh of the object
        
    Returns:
        list: List of top faces
    """
    # Get top faces
    max_z = 0
    top_faces = []
    z_coords = [v.co.z for v in mesh.vertices]
    max_z = max(z_coords)
    for face in mesh.polygons:
        for vert_idx in face.vertices:
            if mesh.vertices[vert_idx].co.z == max_z:
                max_z = mesh.vertices[vert_idx].co.z
                top_faces.append(face)
    return top_faces

def update_uv_coordinates(scene):
    # Get all objects that start with "box"
    box_objects = [obj for obj in scene.objects if obj.name.startswith("box")]

    # Sort them based on the z-coordinate value
    sorted_box_objects = sorted(box_objects, key=lambda obj: obj.matrix_world.to_translation().z, reverse=True)

    # Get the 25 topmost box objects
    topmost_box_objects = sorted_box_objects[:25]

    for i, obj in enumerate(sorted_box_objects):
        mesh = None
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = obj.evaluated_get(depsgraph)
        # mesh = evaluated_obj.to_mesh()
        mesh = obj.data
        mesh.update()
        uv_layer = obj.data.uv_layers.active.data
            

        if obj in topmost_box_objects:
            # Find top faces based on top vertex
            top_faces = get_top_faces(mesh)

            green_faces, red_faces = classify_faces(top_faces)

            update_face_color(green_faces, uv_layer, face_coord_gt_green)
            update_face_color(red_faces, uv_layer, face_coord_gt_red)

        else:

            update_face_color(mesh.polygons, uv_layer, face_coord_gt_red)
            print(f"box: {obj.name}: set to full red")
        
        mesh.update()
        

        # obj.data.keyframe_insert('uv_layers', frame=frame_num)

    # Set the keyframe in Blender
    # bpy.context.scene.update()
    print("\nDone\n")


bpy.app.handlers.frame_change_post.append(update_uv_coordinates)

for frame in range(10):
    bpy.context.scene.frame_set(frame+1)  # set the current frame (frame does not 0-index)
    bpy.context.view_layer.update()  # update the scene
# Get box 6
box = bpy.data.objects["box.006"]
# Get top face normal
mesh = box.data
top_faces = []

# Get top faces
max_z = 0
for face in mesh.polygons:
    for vert_idx in face.vertices:
        if mesh.vertices[vert_idx].co.z == max_z:
            max_z = mesh.vertices[vert_idx].co.z
            top_faces.append(face)
        elif mesh.vertices[vert_idx].co.z > max_z:
            max_z = mesh.vertices[vert_idx].co.z
            top_faces = []
            top_faces.append(face)

# Get angle of top_faces
for face in top_faces:
    # Get the normal of the face
    normal = face.normal

    angle = math.acos(normal.dot((0, 0, 1))) # Calculate the angle relative to the XY plane
    print(f"Box: {box.name} angle: {angle} radians")
    
print("asdasd")