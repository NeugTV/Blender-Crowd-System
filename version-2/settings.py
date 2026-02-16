try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    
import BlenderKI
import controller

#obj_info_dict = {
#    "<Objektname>": {
#        "obj": <Blender Object>,
#        "keyframes_dict": { ... },
#        "state_dict": { ... },
#        "variablen": { ... },
#        "konstrukt_code": { ... }
#    },
#    "<Objektname2>": {
#        ...
#    },
#    ...
#}
obj_info_dict = {}

#gibt für global log txt den aktuellen path des A* aus
path = []

# Globale Einstellungen
def get_fps():
    """FPS der aktuellen Szene zurückgeben."""
    if controller.external_scene_data is not None:
        fps = controller.external_scene_data["fps"]
        return fps
    else:
        return bpy.bpy.context.scene.render.fps
