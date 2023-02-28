import bpy
import numpy as np
import math
from mathutils import Vector

# TODO: Make raycast direction = normal of top face

face_coord_gt_red = np.array([[0.00, 0.00], [0.0, 1.00], [0.50, 1.00], [0.50, 0.00]])
face_coord_gt_green = np.array([[0.5, 0.00], [0.5, 1.00], [1.00, 1.00], [1.00, 0.00]])

OBJECT_ANGLE_SUCTION_THRESHOLD = 2*math.pi/8


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
        for i, vert_idx in enumerate(face.vertices):
        # for vert_idx in range(vert_indices):
            uv_layer[vert_idx].uv = color[i] # Set all uv coordinates to green

def classify_faces(faces, rotation = None):
    """Classify faces as green or red (graspable or not graspable)

    Args:
        faces (bpy.object.data.polygon): Only expecting 8 faces

    Returns:
        touple: Two list of faces, green and red
    """
    green_faces = []
    red_faces = []
    # Find the angle of each faces
    print(f"from classify_faces: len(faces): {len(faces)}")
    for face in faces:
        normal = face.normal
        # Get the normal of the face
        if rotation is not None:
            normal = rotation @ normal
        print(f"normal: {normal}")

        # angle = math.acos(normal.dot((0, 0, 1))) # Calculate the angle relative to the XY plane
        angle = normal.angle(Vector((0, 0, 1)))
        print(f"angle: {angle}")
        # if obj.name == "box.086":
        #     print(f"Box: {obj.name} face: {face.index} angle: {angle} radians")
        if -OBJECT_ANGLE_SUCTION_THRESHOLD < angle < OBJECT_ANGLE_SUCTION_THRESHOLD:
            green_faces.append(face)
        else:
            print(f"not in threshold was: {angle}")
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
    color_str = "green" if color[0][0] == 0.5 else "red"
    print(f"len(faces): {len(faces)} color: {color_str}")
    vert_indices = 4 # Only expect to be 4 vertices per face
    for face in faces:
        for vert_idx in range(vert_indices):
            uv_layer[face.loop_start + vert_idx].uv = color[vert_idx] # Set all uv coordinates to green

def get_top_faces(mesh, transformation = None):
    """Get the top faces of the mesh
    TODO: does not seems to work?
    Args:
        mesh (bpy.object.data): Mesh of the object
        
    Returns:
        list: List of top faces
    """
    # Get top faces
    max_z = 0
    top_faces = []
    vertecise = mesh.vertices

    
    z_coords = None
    if transformation is not None:
        z_coords = [(v.co @ transformation).z for v in vertecise]
    else:
        z_coords = [v.co.z for v in vertecise]
    
    max_z = max(z_coords)
    print(f"from get_top_faces: len(mesh.polygons): {len(mesh.polygons)}")
    for face in mesh.polygons:
        for vert_idx in face.vertices:
            co = None
            if transformation is not None:
                co = (vertecise[vert_idx].co @ transformation)
            else:
                co = (vertecise[vert_idx].co)

            if co.z == max_z:
                top_faces.append(face)
                print(f"face: {face.index} max_z: {max_z}")
                break

    return top_faces

def main():
    # Get all objects that start with "box"
    box_objects = [obj for obj in bpy.data.objects if obj.name.startswith("box")]

    # Get scene
    scene = bpy.context.scene

    # Get depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()


    for box in box_objects:
        mesh = box.data
        uv_layer = box.data.uv_layers.active.data

        # Get the evaluated object from the depsgraph
        evaluated_object = box.evaluated_get(depsgraph)

        # Get the location of the evaluated object
        loc = evaluated_object.matrix_world.translation
        rot = evaluated_object.matrix_world.to_3x3()
        transform = evaluated_object.matrix_world

        hit, pos, normal, face_id, obj, matrix = scene.ray_cast(depsgraph, loc, (0,0,1)) 

        i = 0
        offset = Vector((0, 0, 0.001))
        while hit and obj.name == box.name:
            hit, pos, normal, face_id, obj, matrix = scene.ray_cast(depsgraph, pos, (0,0,1))
            i += 1
            pos = pos + offset
            if i > 100:
                print(f"box: {box.name}: hit 100 times")
                break

        # start with all face red
        update_face_color(mesh.polygons, uv_layer, face_coord_gt_red)

        if not hit:
            # Find top faces based on top vertex
            # top_faces = get_top_faces(mesh, transform)
            green_faces, red_faces = classify_faces(mesh.polygons, rot)

            update_face_color(green_faces, uv_layer, face_coord_gt_green)
            update_face_color(red_faces, uv_layer, face_coord_gt_red)    

if __name__ == "__main__":
    main()