import bpy
import numpy as np
import math
from mathutils import Vector
import copy
import threading

face_coord_gt_red = np.array([[0.00, 0.00], [0.0, 1.00], [0.50, 1.00], [0.50, 0.00]])
face_coord_gt_green = np.array([[0.5, 0.00], [0.5, 1.00], [1.00, 1.00], [1.00, 0.00]])

OBJECT_ANGLE_SUCTION_THRESHOLD = 2*math.pi/8

class FaceColor:
    def __init__(self, box_objects_array):
        # Get all objects that start with "box"
        self.box_objects = box_objects_array

        self.box_original_uv_layer_values = []
        for obj in self.box_objects:
            uv_layer = obj.data.uv_layers.active.data
            uv_layer_values = []
            for face in obj.data.polygons:
                uv_values = [copy.copy(uv_layer[face.loop_start + vert_idx].uv) for vert_idx in range(4)]
                uv_layer_values.append(uv_values)
            self.box_original_uv_layer_values.append(uv_layer_values)
        # self.box_original_uv_layer = [obj.data.uv_layers.active.data for obj in self.box_objects]

        # Get scene
        self.scene = bpy.context.scene

        # Get depsgraph
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        self.changed = False
    # def update_face_color(self, faces, uv_layer, color):
    #     """Update the color of the faces
        
    #     Args:
    #         faces (bpy.object.data.polygon): Only expecting 8 faces
    #         uv_layer (obj.data.uv_layers.active.data): Uv layer of the object
    #         color (np.array): Color to set the faces to
            
    #     Returns:
    #         None
    #     """
    #     vert_indices = 4 # Only expect to be 4 vertices per face
    #     for face in faces:
    #         for i, vert_idx in enumerate(face.vertices):
    #         # for vert_idx in range(vert_indices):
    #             uv_layer[vert_idx].uv = color[i] # Set all uv coordinates to green

    def classify_faces_strict(self, faces, obj):
        """Classify faces as green or red (graspable or not graspable)

        Args:
            faces (bpy.object.data.polygon): Only expecting 8 faces

        Returns:
            touple: Two list of faces, green and red
        """
        obj_center = obj.evaluated_get(self.depsgraph).matrix_world.translation
        rotation = obj.evaluated_get(self.depsgraph).matrix_world.to_3x3()
        hit_obj = obj
        distance = 0.5
        green_faces = []
        red_faces = []

        def ray_cast(location, direction, hit_result, hit_obj_result, index):
            hit, pos, normal, face_id, hit_obj, matrix = self.scene.ray_cast(self.depsgraph, location, direction, distance=distance)
            i = 0
            offset = normal * 0.01 # move 1 cm in the direction of the normal
            while i < 100 and hit_obj is obj:
                hit, pos, normal, face_id, hit_obj, matrix = self.scene.ray_cast(self.depsgraph, pos+offset, direction, distance=distance)
                pos += normal * 0.01 # move 1 cm in the direction of the normal
                i += 1
            hit_result[index] = hit
            hit_obj_result[index] = hit_obj
        
        for face in faces:
            i = 0
            normal = face.normal
            offset = normal * 0.01
            thread_array = []
            hit_result = [False] * 4
            hit_obj_result = [None] * 4
            hit = False
            # Find 4 edges of the face
            for vert_idx in range(4):
                vert = obj.data.vertices[face.vertices[vert_idx]]
                vert_world = obj.matrix_world @ vert.co
                direction = (vert_world - obj_center).normalized()
                thread_array.append(threading.Thread(target=ray_cast, args=(vert_world, (0,0,1), hit_result, hit_obj_result, i)))
                thread_array[-1].start()
            for thread in thread_array:
                thread.join()
            for result in hit_result:
                if result:
                    hit = True
                    break                
            if not hit:
                green_faces.append(face)
            else:
                red_faces.append(face)
        return green_faces, red_faces
    

    def classify_faces_with_normal(self, faces, obj):
        """Classify faces as green or red (graspable or not graspable)

        Args:
            faces (bpy.object.data.polygon): Only expecting 8 faces

        Returns:
            touple: Two list of faces, green and red
        """
        obj_center = obj.evaluated_get(self.depsgraph).matrix_world.translation
        rotation = obj.evaluated_get(self.depsgraph).matrix_world.to_3x3()
        hit_obj = obj
        distance = 0.5
        green_faces = []
        red_faces = []

        for face in faces:
            i = 0
            normal = face.normal
            offset = normal * 0.01
            hit = False

            # Get the evaluated object from the depsgraph
            evaluated_object = obj.evaluated_get(self.depsgraph)

 
            normal = rotation @ normal


            hit, pos, normal, face_id, hit_obj, matrix = self.scene.ray_cast(self.depsgraph, obj_center, normal, distance=distance)
            i = 0
            offset = normal * 0.01 # move 1 cm in the direction of the normal
            while i < 100 and hit_obj is obj:
                hit, pos, normal, face_id, hit_obj, matrix = self.scene.ray_cast(self.depsgraph, pos+offset, normal, distance=distance)
                pos += normal * 0.01 # move 1 cm in the direction of the normal
                i += 1

            if not hit:
                green_faces.append(face)
            else:
                red_faces.append(face)
        return green_faces, red_faces

    def classify_faces(self, faces, obj):
        """Classify faces as green or red (graspable or not graspable)

        Args:
            faces (bpy.object.data.polygon): Only expecting 8 faces

        Returns:
            touple: Two list of faces, green and red
        """
        # Get the evaluated object from the depsgraph
        evaluated_object = obj.evaluated_get(self.depsgraph)

        # Get the location of the evaluated object
        loc = evaluated_object.matrix_world.translation
        rot = evaluated_object.matrix_world.to_3x3()
        transform = evaluated_object.matrix_world

        hit, pos, normal, face_id, obj, matrix = self.scene.ray_cast(self.depsgraph, loc, (0,0,1)) 

        i = 0
        offset = Vector((0, 0, 0.001))
        while hit and obj.name == obj.name:
            hit, pos, normal, face_id, obj, matrix = self.scene.ray_cast(self.depsgraph, pos, (0,0,1))
            i += 1
            pos = pos + offset
            if i > 100:
                print(f"box: {obj.name}: hit 100 times")
                break

        green_faces = []
        red_faces = []
        # Find the angle of each faces
        # print(f"from classify_faces: len(faces): {len(faces)}")
        for face in faces:
            normal = face.normal
            # Get the normal of the face
            normal = rot @ normal
            # print(f"normal: {normal}")

            # angle = math.acos(normal.dot((0, 0, 1))) # Calculate the angle relative to the XY plane
            angle = normal.angle(Vector((0, 0, 1)))
            # print(f"angle: {angle}")
            # if obj.name == "box.086":
            #     print(f"Box: {obj.name} face: {face.index} angle: {angle} radians")
            if -OBJECT_ANGLE_SUCTION_THRESHOLD < angle < OBJECT_ANGLE_SUCTION_THRESHOLD:
                green_faces.append(face)
            else:
                # print(f"not in threshold was: {angle}")
                red_faces.append(face)
        return green_faces, red_faces


    def update_face_color(self, faces, uv_layer, color):
        """Update the color of the faces
        
        Args:
            faces (bpy.object.data.polygon): Only expecting 8 faces
            uv_layer (obj.data.uv_layers.active.data): Uv layer of the object
            color (np.array): Color to set the faces to
            
        Returns:
            None
        """
        # color_str = "green" if color[0][0] == 0.5 else "red"
        # print(f"len(faces): {len(faces)} color: {color_str}")
        vert_indices = 4 # Only expect to be 4 vertices per face
        for face in faces:
            for vert_idx in range(vert_indices):
                uv_layer[face.loop_start + vert_idx].uv = color[vert_idx] # Set all uv coordinates to green


    def get_top_faces(self, mesh, transformation = None):
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
        # print(f"from get_top_faces: len(mesh.polygons): {len(mesh.polygons)}")
        for face in mesh.polygons:
            for vert_idx in face.vertices:
                co = None
                if transformation is not None:
                    co = (vertecise[vert_idx].co @ transformation)
                else:
                    co = (vertecise[vert_idx].co)

                if co.z == max_z:
                    top_faces.append(face)
                    # print(f"face: {face.index} max_z: {max_z}")
                    break

        return top_faces

    def paint_gt(self):
        """Paint the ground truth on the objects
        """
        self.changed = True
        for box in self.box_objects:
            mesh = box.data
            uv_layer = box.data.uv_layers.active.data

            # Get the evaluated object from the depsgraph
            evaluated_object = box.evaluated_get(self.depsgraph)

            green_faces, red_faces = self.classify_faces_strict(mesh.polygons, box)
            self.update_face_color(green_faces, uv_layer, face_coord_gt_green)
            self.update_face_color(red_faces, uv_layer, face_coord_gt_red)    

    def return_face_texture(self):
        """Return the texture of the objects to its original state
        """
        self.changed = False
        vert_indices = 4 # Only expect to be 4 vertices per face
        for obj, uv_data_values in zip(self.box_objects, self.box_original_uv_layer_values):
            uv_layer = obj.data.uv_layers.active.data
            for face, uv_data in zip(obj.data.polygons, uv_data_values):
                for vert_idx in range(vert_indices):
                    uv_layer[face.loop_start + vert_idx].uv = uv_data[vert_idx]
        

def main():
    box = bpy.data.objects["panodil_zap.002"]
    print(f"--------box: {box.name}--------")
    box_arr = [box]
    fc = FaceColor(box_arr)
    fc.paint_gt()

if __name__ == "__main__":
    main()