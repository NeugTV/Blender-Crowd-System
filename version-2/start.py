import sys
import os
import json
import time

# Prüfen ob wir in Blender laufen
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# Lokaler Pfad
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import settings
import controller
import BlenderKI


# ----------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------

def export_scene_to_json():
    """Liest alle relevanten Daten aus Blender und speichert sie in scene_data.json."""
    if not IN_BLENDER:
        print("Fehler: Schritt 1 muss in Blender ausgeführt werden.")
        return

    print("Lese Blender-Daten...")

    data = {}

    # 1. Deine bisherigen Funktionen
    BlenderKI.check_prop()
    BlenderKI.get_state_txt_files()

    data["obj_info_dict"] = settings.obj_info_dict

    # 2. Geometrie aller Objekte
    scene_objects = []
    for obj in bpy.data.objects:
        
        scene_objects.append({
            "name": obj.name,
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "type": obj.type,
            "obj_width": BlenderKI.get_obj_width(obj)
        })

    data["objects"] = scene_objects
    
    # 3. Frame-Infos
    scene = bpy.context.scene
    data["frame_start"] = scene.frame_start
    data["frame_end"] = scene.frame_end
    data["fps"] = bpy.context.scene.render.fps
    data["current_frame"] = scene.frame_start
    
    # 4. verices object
    vertices_list_obj_dict = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            vertices_list_obj_dict[obj.name] = BlenderKI.get_vertices_worldspace(obj)
    data["vertices"] = vertices_list_obj_dict
    
    # 5 bbox
    bbox_world_dict = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bbox_world_dict[obj.name] = BlenderKI.world_bbox_corners(obj)
    data["bbox"] = bbox_world_dict

    # Speichern
    out_path = os.path.join(BASE_DIR, "scene_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

    print("✔ Szene erfolgreich exportiert nach scene_data.json")


def run_calculation():
    """Berechnet Keyframes ohne Blender (CMD)."""
    print("Lade scene_data.json...")

    json_path = os.path.join(BASE_DIR, "scene_data.json")
    if not os.path.exists(json_path):
        print("Fehler: scene_data.json nicht gefunden. Bitte Schritt 1 zuerst ausführen.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Übergabe der Daten an controller
    controller.external_scene_data = data

    start = data["frame_start"]
    end = data["frame_end"]

    print(f"Starte Berechnung von Frame {start} bis {end}...")
    
    settings.obj_info_dict = controller.external_scene_data["obj_info_dict"]
    
    t0 = time.time()
    controller.controller_durchlauf(start, end)
    t1 = time.time()

    print(f"✔ Berechnung abgeschlossen in {round(t1 - t0, 2)} Sekunden")
    print("✔ Keyframes wurden in TXT gespeichert")


def import_keyframes():
    """Importiert Keyframes in Blender und setzt F-Curves linear."""
    if not IN_BLENDER:
        print("Fehler: Schritt 3 muss in Blender ausgeführt werden.")
        return

    print("Importiere Keyframes...")

    controller.apply_keyframes_from_txt()
    controller.set_all_fcurves_linear()

    print("✔ Keyframes erfolgreich importiert und F-Curves gesetzt")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        try:
            schritt = int(sys.argv[-1])
        except:
            schritt = 0
    else:
        schritt = 0

    print(f"Starte Schritt {schritt}...")

    if schritt == 1:
        export_scene_to_json()

    elif schritt == 2:
        run_calculation()

    elif schritt == 3:
        import_keyframes()

    else:
        print("Ungültiger Schritt. Nutze:")
        print("  python start.py 1   # Blender → JSON exportieren")
        print("  python start.py 2   # CMD → Berechnung durchführen")
        print("  python start.py 3   # Blender → Keyframes importieren")
