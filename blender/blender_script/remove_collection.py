import bpy

objects = bpy.data.collections['copies'].all_objects
# Delete all objects that have the name "box" excluding the original box
for obj in objects:
    bpy.data.objects.remove(obj)