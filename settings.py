import bpy

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

# Dynamische Objekte (z.B. für Navigation / A*)
dynamic_obj_dict = {}

# Zwischenspeicher für Wegpunkte / Pfade
neuer_point_dict = {}

#gibt für global log txt den aktuellen path des A* aus
path = []

# Globale Einstellungen
def get_scene():
    """Hilfsfunktion: aktuelle Szene zurückgeben."""
    return bpy.context.scene

def get_fps():
    """FPS der aktuellen Szene zurückgeben."""
    return get_scene().render.fps
