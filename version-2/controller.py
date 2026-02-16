# controller.py
# Steuert pro Frame die States, Bewegung, Animation und Keyframes

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


import os, sys
from mathutils import Vector
from math import radians
import math
import mathutils
import heapq
import os, json

external_scene_data = None

import settings
import BlenderKI

import datetime

TURN_DURATION = 0.25      # Sekunden für Drehung zum nächsten Wegpunkt

LOGFILE = "blenderki_debug_log.txt"
KEYFRAME_FILE = "blenderki_keyframes.txt"

def log(msg):
    """Schreibt Debug-Informationen in eine TXT-Datei."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
        
def write_keyframes(obj_name, frame, location=None, rotation=None):
    if location is not None:
        if "keyframes_dict" in settings.obj_info_dict[obj_name].keys():
            if "location" in settings.obj_info_dict[obj_name]["keyframes_dict"].keys():
                if frame in settings.obj_info_dict[obj_name]["keyframes_dict"]["location"].keys():
                    settings.obj_info_dict[obj_name]["keyframes_dict"]["location"][frame] = location
                else:
                    settings.obj_info_dict[obj_name]["keyframes_dict"]["location"] = {frame: location}
            else:
                settings.obj_info_dict[obj_name]["keyframes_dict"] = {"location": {frame: location}}
        else:
            settings.obj_info_dict[obj_name] = {"keyframes_dict": {"location": {frame: location}}}
    if rotation is not None:
        if "keyframes_dict" in settings.obj_info_dict[obj_name].keys():
            if "rotation" in settings.obj_info_dict[obj_name]["keyframes_dict"].keys():
                if frame in settings.obj_info_dict[obj_name]["keyframes_dict"]["rotation"].keys():
                    settings.obj_info_dict[obj_name]["keyframes_dict"]["rotation"][frame] = rotation
                else:
                    settings.obj_info_dict[obj_name]["keyframes_dict"]["rotation"] = {frame: rotation}
            else:
                settings.obj_info_dict[obj_name]["keyframes_dict"] = {"rotation": {frame: rotation}}
        else:
            settings.obj_info_dict[obj_name] = {"keyframes_dict": {"rotation": {frame: rotation}}}
    
"""   
def write_keyframes(obj, frame, location=None, rotation=None):
    Schreibt Keyframe-Informationen in eine TXT-Datei.
    dateiAlt = ""
    dateiNeu = ""
    trennung1 = '***'
    trennung2 = '###'
    trennung3 = '|'
    trennung4 = '---'
    trennung5 = '$'
    objektDict = {} #{objektName: {frameNummer: {'location': loc_Vector, 'rotation': rot_Vector}}}

    # Einlesen der bestehenden Keyframe-Daten aus der Datei und Umwandeln in ein Dictionary
    with open(KEYFRAME_FILE, "r", encoding="utf-8") as f:
        dateiAlt = f.read()
        if trennung1 in dateiAlt or dateiAlt.strip() != "": 
            teileNachObjekt = []
            if trennung1 in dateiAlt:
                teileNachObjekt = dateiAlt.split(trennung1)
            else:
                teileNachObjekt = [dateiAlt]
            for teil in teileNachObjekt:
                if trennung2 in teil:
                    objektName, rest = teil.split(trennung2, 1)
                    if trennung3 in rest:
                        frameInfos = rest.split(trennung3)
                        frameDict = {}
                        for frameInfo in frameInfos:
                            if trennung4 in frameInfo:
                                frameNummer, locRotStr = frameInfo.split(trennung4, 1)
                                locRotDict = {}
                                if trennung5 in locRotStr:
                                    # Wenn sowohl Location als auch Rotation vorhanden sind, zuerst splitten und dann separat parsen
                                    locStr, rotStr = locRotStr.split(trennung5, 1)
                                    if locStr != "":
                                        locArray = locStr.replace("Location: ", "").strip().split(",")
                                        locRotDict['location'] = mathutils.Vector((float(locArray[0]), float(locArray[1]), float(locArray[2])))
                                    if rotStr != "":
                                        rotArray = rotStr.replace("Rotation: ", "").strip().split(",")
                                        locRotDict['rotation'] = mathutils.Euler((float(rotArray[0]), float(rotArray[1]), float(rotArray[2])), 'XYZ')
                                    
                                else:
                                    # Wenn nur Location oder Rotation vorhanden ist, entsprechend parsen
                                    if "Location: " in locRotStr:
                                        locRotArray = locRotStr.replace("Location: ", "").strip().split(",")
                                        locRotDict['location'] = mathutils.Vector((float(locRotArray[0]), float(locRotArray[1]), float(locRotArray[2])))
                                    elif "Rotation: " in locRotStr:
                                        locRotArray = locRotStr.replace("Rotation: ", "").strip().split(",")
                                        locRotDict['rotation'] = mathutils.Euler((float(locRotArray[0]), float(locRotArray[1]), float(locRotArray[2])), 'XYZ')
                                frameDict[int(frameNummer.strip())] = locRotDict
                        objektDict[objektName.strip()] = frameDict
    
    # Aktualisieren des Dictionaries mit den neuen Keyframe-Informationen            
    if obj.name in objektDict:
        if frame in objektDict[obj.name]:
            if location is not None:
                objektDict[obj.name][frame]['location'] = location
            if rotation is not None:
                objektDict[obj.name][frame]['rotation'] = rotation
        else:
            objektDict[obj.name][frame] = {}
            if location is not None:
                objektDict[obj.name][frame]['location'] = location
            if rotation is not None:
                objektDict[obj.name][frame]['rotation'] = rotation
    else:
        objektDict[obj.name] = {}
        objektDict[obj.name][frame] = {}
        if location is not None:
            objektDict[obj.name][frame]['location'] = location
        if rotation is not None:
            objektDict[obj.name][frame]['rotation'] = rotation
    
    # Umwandeln des aktualisierten Dictionaries zurück in den Datei-String     
    
    izahl1 = 0       
    for objName, frameDict in objektDict.items():
        if izahl1 == 0:
            dateiNeu += f"{objName}{trennung2}"
            izahl1 += 1
        else:
            dateiNeu += f"{trennung1}{objName}{trennung2}"
        for frameNummer, locRotDict in sorted(frameDict.items()):
            locStr = f"Location: {locRotDict['location'].x:.4f},{locRotDict['location'].y:.4f},{locRotDict['location'].z:.4f}" if 'location' in locRotDict else ""
            rotStr = f"Rotation: {locRotDict['rotation'].x:.4f},{locRotDict['rotation'].y:.4f},{locRotDict['rotation'].z:.4f}" if 'rotation' in locRotDict else ""
            locRotStr = locStr + (trennung5 + rotStr if rotStr else "")
            dateiNeu += f"{frameNummer}{trennung4}{locRotStr}{trennung3}"
        dateiNeu += "\n"
    
    # Schreiben der aktualisierten Keyframe-Daten zurück in die Datei
    with open(KEYFRAME_FILE, "w", encoding="utf-8") as f:
        f.write(dateiNeu)
 """   
       
# ----------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------
def frequenz(freq):
    if external_scene_data is not None and IN_BLENDER is None:
        currentFrame = external_scene_data["current_frame"]
    else:
        currentFrame = bpy.data.scenes[0].frame_current
    if freq == 0:
        return True
    return currentFrame % freq == 0

def state_sensor(objName, stateIndex, freq=0):
    if not frequenz(freq):
        return False
    return settings.obj_info_dict[objName]['state_dict'][stateIndex]['state_name'] == 'start'

def variable_sensor(type, value, freq=0):
    if not frequenz(freq):
        return None
    if type == 'int':
        return int(value)
    elif type == 'float':
        return float(value)
    elif type == 'string':
        return str(value)
    elif type == 'bool':
        return bool(value)
    
    

def distanz_sensor(objName, zielObjName, distValue, freq=0):
    if not frequenz(freq):
        return False
    
    p1 = bpy.data.objects[objName].location
    p2 = bpy.data.objects[zielObjName].location
    dist = (p2 - p1).length
    return dist <= abs(distValue)

def raycast_sensor(obj, zielName, direction, distance, freq=0):
    if not frequenz(freq):
        return False

    # Blender-Version
    if IN_BLENDER and external_scene_data is None:
        origin = obj.location
        result, loc, normal, index, obj2, matrix = bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph, origin, direction
        )
        return result and obj2.name == zielName and (origin - loc).length <= distance

    # Headless-Version (Schritt 2)
    # → stark vereinfachter Raycast: prüft nur Distanz + Richtung
    origin = obj["location"]
    target = None

    for o in external_scene_data["objects"]:
        if o["name"] == zielName:
            target = o
            break

    if target is None:
        return False

    target_pos = target["location"]

    # Richtung prüfen (Skalarprodukt)
    vec = [
        target_pos[i] - origin[i]
        for i in range(3)
    ]

    dot = vec[0]*direction[0] + vec[1]*direction[1] + vec[2]*direction[2]
    if dot <= 0:
        return False

    # Distanz prüfen
    dist = sum((vec[i])**2 for i in range(3)) ** 0.5
    return dist <= distance

def delay_sensor(delayFrames, startFrame, loopOrOnce):
    # Blender-Version
    if IN_BLENDER and external_scene_data is None:
        currentFrame = bpy.context.scene.frame_current
    else:
        currentFrame = external_scene_data["current_frame"]

    if loopOrOnce:
        return currentFrame >= startFrame + delayFrames

    return currentFrame >= startFrame and (currentFrame - startFrame) % delayFrames == 0

def dict_veringerung(dictKopie, xZahl=5):
    neuesDict = {}
    d = 0
    for i, (index, item) in enumerate(dictKopie.items()):
        if index == 0 or index == len(dictKopie) - 1 or index % xZahl == 0:
            neuesDict[d] = item
            d += 1
    return neuesDict





def camera_follow_objects(cam_name, obj_list, distance=5.0):
    cam = bpy.data.objects[cam_name]
    center = Vector((0,0,0))
    for obj in obj_list:
        center += bpy.data.objects[obj].location
    center /= len(obj_list)
    cam.location = center + Vector((0, -distance, distance))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    frame = bpy.context.scene.frame_current
    cam.keyframe_insert(data_path="location", frame=frame)
    cam.keyframe_insert(data_path="rotation_euler", frame=frame)
    

    
def build_frame_dict_from_path_time(world_path, start_frame, zeit_in_sek):
    """Verteilt die Wegpunkte gleichmäßig über eine gegebene Zeit."""
    fps = settings.get_fps()
    total_frames = max(1, int(zeit_in_sek * fps))
    result = {}
    if len(world_path) == 0:
        return result
    if len(world_path) == 1:
        result[start_frame] = {'wegpunkt': world_path[0]}
        return result

    frames_per_segment = total_frames / (len(world_path) - 1)
    for i, pos in enumerate(world_path):
        frame = int(start_frame + i * frames_per_segment)
        result[frame] = {'wegpunkt': pos}
    return result


def face_direction(obj, target_pos):
    if external_scene_data is not None and IN_BLENDER is None:
        target = next((o for o in external_scene_data["objects"] if o["name"] == obj), None)
        direction = (target_pos - mathutils.Vector((target["location"][0], target["location"][1], target["location"][2]))).normalized()
    else:
        direction = (target_pos - obj.location).normalized()
    rot = direction.to_track_quat('Y', 'Z').to_euler()
    return rot



def look_at_angle(from_pos, to_pos):
    """
    Berechnet den Winkel (in Radiant) um Z, damit das Objekt von from_pos nach to_pos schaut.
    """
    direction = Vector((to_pos.x - from_pos.x, to_pos.y - from_pos.y))
    if direction.length == 0:
        return None
    angle = math.atan2(direction.y, direction.x)
    return angle


def animate_agent(obj, world_path, fps, start_frame, speed_cm_s, zeitAktiv, zeit_in_sek):
    rotDict = {}
    moveDict = {}
    rotZeitDict = {}
    moveZeitDict = {}

    # Wenn weniger als 2 gültige Punkte → keine Bewegung
    if len(world_path) < 2:
        return moveDict, rotDict

    # Gesamtdistanz berechnen
    total_dist = 0.0
    for i in range(1, len(world_path)):
        p0, _ = world_path[i-1]
        p1, _ = world_path[i]
        total_dist += (p1 - p0).length

    if total_dist == 0:
        return moveDict, rotDict

    # Zeit aus Geschwindigkeit
    total_time = total_dist / (speed_cm_s / 100.0)
    time_per_meter = total_time / total_dist

    current_frame = start_frame
    
    turn_end_frame_last = current_frame
    last_rot = None

    for i in range(1, len(world_path)):
        p0, t0 = world_path[i-1]
        p1, t1 = world_path[i]

        segment_dist = (p1 - p0).length
        segment_time = segment_dist * time_per_meter
        segment_frames = max(1, int(segment_time * fps))
        
        
        if len(rotDict) == 0:
            # Erste Rotation direkt setzen
            rotDict[current_frame] = face_direction(obj, p1)
        if len(moveDict) == 0:
            # Erste Position direkt setzen
            moveDict[current_frame] = p0
        

        # Drehung zum nächsten Wegpunkt
        angle = look_at_angle(p0, p1)
        turn_frames = max(1, int(round(TURN_DURATION * settings.get_fps())))
        turn_end_frame = current_frame + turn_frames

        rot = mathutils.Euler((0.0, 0.0, angle), 'XYZ')
        
        turn_frames = int(TURN_DURATION * fps)

        # Rotation eintragen
        if last_rot is not None:
            rotDict[turn_end_frame_last] = last_rot
            last_rot = rot
            turn_end_frame_last = current_frame
        rotDict[turn_end_frame] = rot

        # Position am Ende des Segments
        moveDict[current_frame + segment_frames] = p1

        current_frame += segment_frames
        
    if zeitAktiv:
        frameSpanne = current_frame - start_frame
        gewünschteSpanne = int(zeit_in_sek * fps)
        for frame in moveDict:
            new_frame = start_frame + int((frame / frameSpanne) * gewünschteSpanne)
            moveZeitDict[new_frame] = moveDict[frame]
        for frame in rotDict:
            new_frame = start_frame + int((frame / frameSpanne) * gewünschteSpanne)
            rotZeitDict[new_frame] = rotDict[frame]
        return moveZeitDict, rotZeitDict
    else:
        return moveDict, rotDict


def objekt_folgen_vorher(obj_name, pointDict, start_frame, start_loc, ziel_loc, zeit_in_sek, konstant_speed=False):
    fps = settings.get_fps()
    total_frames = int(zeit_in_sek * fps)
    result = {}
    frames_per_point = max(1, int(total_frames / max(1, len(pointDict) - 1)))
    current_frame = start_frame
    for i in range(len(pointDict)):
        result[current_frame] = {'wegpunkt': pointDict[i]}
        current_frame += frames_per_point
    return result

def calculate_speed_path(start, ziel, speed_cm_s, start_frame):
    fps = settings.get_fps()
    dist = (ziel - start).length
    total_frames = int((dist / speed_cm_s) * fps)
    result = {}
    for i in range(total_frames + 1):
        t = i / total_frames
        pos = start.lerp(ziel, t)
        result[start_frame + i] = {'wegpunkt': pos}
    return result

def point_obj_update(obj_name, modus, ziel_vec, zeit_in_sek):
    if external_scene_data is not None and IN_BLENDER is None:
        target = next((o for o in external_scene_data["objects"] if o["name"] == obj_name), None)
        loc = mathutils.Vector((target["location"][0], target["location"][1], target["location"][2]))
    else:
        loc = settings.obj_info_dict[obj_name]['obj'].location
        
    if modus in ['lookAt', 'startTrack']:
        direction = ziel_vec - loc
        if external_scene_data is not None and IN_BLENDER is None:
            target1 = next((o for o in external_scene_data["objects"] if o["name"] == obj_name), None)
            target1["rotation_euler"] = direction.to_track_quat('Z', 'Y').to_euler()
        else:
            settings.obj_info_dict[obj_name]['obj'].rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

def set_rotation_to_target(obj_name, ziel_name):
    if external_scene_data is not None and IN_BLENDER is None:
        target1 = next((o for o in external_scene_data["objects"] if o["name"] == obj_name), None)
        target2 = next((o for o in external_scene_data["objects"] if o["name"] == ziel_name), None)
        direction = target2["location"] - target1["location"]
        target1["rotation_euler"] = direction.to_track_quat('Z', 'Y').to_euler()
    else:
        obj = bpy.data.objects[obj_name]
        ziel = bpy.data.objects[ziel_name]
        direction = ziel.location - obj.location
        obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

def play_animation(obj_name, anim_name, cut_von, cut_bis, loop_or_once):
    # Headless: Animationen existieren nicht
    if external_scene_data is not None:
        return  # Keine Aktion in Schritt 2 (WICHTIG: VERÄNDERN)

    # Blender-Version
    obj = bpy.data.objects[obj_name]
    if obj.animation_data is None:
        obj.animation_data_create()
    obj.animation_data.action = bpy.data.actions.get(anim_name)
    # Schnitt und Loop nicht direkt steuerbar, müsste über NLA oder manuell erfolgen

def apply_keyframes_from_txt():
    
    trennung1 = '***'
    trennung2 = '###'
    trennung3 = '|'
    trennung4 = '---'
    trennung5 = '$'
    
    with open(KEYFRAME_FILE, "r", encoding="utf-8") as f:
        dateiAlt = f.read()
        if trennung1 in dateiAlt: 
            teileNachObjekt = dateiAlt.split(trennung1)
            for teil in teileNachObjekt:
                if trennung2 in teil:
                    objektName, rest = teil.split(trennung2, 1)
                    if trennung3 in rest:
                        frameInfos = rest.split(trennung3)
                        for frameInfo in frameInfos:
                            if trennung4 in frameInfo:
                                frameNummer, locRotStr = frameInfo.split(trennung4, 1)
                                locRotDict = {}
                                if trennung5 in locRotStr:
                                    locStr, rotStr = locRotStr.split(trennung5, 1)
                                    if locStr != "":
                                        x, y, z = locStr.replace("Location: ", "").strip().split(",")
                                        locVec = mathutils.Vector((float(x), float(y), float(z)))
                                        bpy.data.objects[objektName].location = locVec
                                        bpy.data.objects[objektName].keyframe_insert(data_path="location", frame=int(frameNummer.strip()))
                                    if rotStr != "":
                                        x1, y1, z1 = rotStr.replace("Rotation: ", "").strip().split(",")
                                        rotVec = mathutils.Euler((float(x1), float(y1), float(z1)), 'XYZ')
                                        bpy.data.objects[objektName].rotation_euler = rotVec
                                        bpy.data.objects[objektName].keyframe_insert(data_path="rotation_euler", frame=int(frameNummer.strip()))
                                else:
                                    if "Location: " in locRotStr:
                                        x, y, z = locRotStr.replace("Location: ", "").strip().split(",")
                                        locVec = mathutils.Vector((float(x), float(y), float(z)))
                                        bpy.data.objects[objektName].location = locVec
                                        bpy.data.objects[objektName].keyframe_insert(data_path="location", frame=int(frameNummer.strip()))
                                    elif "Rotation: " in locRotStr:
                                        x1, y1, z1 = locRotStr.replace("Rotation: ", "").strip().split(",")
                                        rotVec = mathutils.Euler((float(x1), float(y1), float(z1)), 'XYZ')
                                        bpy.data.objects[objektName].rotation_euler = rotVec
                                        bpy.data.objects[objektName].keyframe_insert(data_path="rotation_euler", frame=int(frameNummer.strip()))
                                        

"""
def apply_keyframes_from_dynamic_dict():
    scene = bpy.context.scene
    
    for obj_name, frame_dict in settings.dynamic_obj_dict.items():
        obj = settings.obj_info_dict[obj_name]['obj']
        for frame, data in frame_dict.items():
            for key, value in data.items():
                if key == 'variante1':
                    log(f"{obj_name} bewegt sich zu {value} frame {frame}")
                    scene.frame_set(frame)
                    obj.location = value
                    obj.keyframe_insert("location")
                if key == 'variante2':
                    log(f"{obj_name} rotiert zu {value} frame {frame}")
                    scene.frame_set(frame)
                    obj.rotation_euler = value
                    obj.keyframe_insert("rotation_euler")
                
               
             
def apply_keyframes_from_dynamic_dict():
    scene = bpy.context.scene

    for obj_name, frame_dict in settings.dynamic_obj_dict.items():

        # Objekt existiert nicht → überspringen
        if obj_name not in bpy.data.objects:
            log(f"Objekt {obj_name} existiert nicht in der Szene – überspringe.")
            continue

        obj = bpy.data.objects[obj_name]

        # Keyframes löschen, damit wir sauber neu schreiben
        if obj.animation_data:
            obj.animation_data_clear()

        # Alle Frames sortiert durchgehen
        for frame in sorted(frame_dict.keys()):
            data = frame_dict[frame]

            # Position
            if 'variante1' in data:
                pos = data['variante1']
                obj.location = pos
                obj.keyframe_insert(data_path="location", frame=frame)

            # Rotation
            if 'variante2' in data:
                rot = data['variante2']
                obj.rotation_euler = rot
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        log(f"Keyframes für {obj_name} erfolgreich gesetzt.")
"""
                
                
def set_all_fcurves_linear():
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.interpolation = 'LINEAR'
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



# ----------------------------------------------------------
# Haupt-Controller
# ----------------------------------------------------------
def controller_durchlauf(startFrame, endFrame):
    log(f"Starte controller_durchlauf: start={startFrame}, end={endFrame}")
    aktiverState = {}
    aktiverStateKopie = {}
    objEinmalAktiv = {}
    zeitDict = {}
    einmalAktivDict = {}
    pointDictDict = {}
    objEinmalKeyIndexDict = {}
    alleAgentenEinmalAktiv = False
    agentenDict = {}
    frameLength = endFrame - startFrame

    for index in settings.obj_info_dict:
        for f in range(startFrame, endFrame):
            settings.obj_info_dict[index]['keyframes_dict'][f] = {}

    for t in range(frameLength):
        current_frame = startFrame + t
        
        if t % 10 == 0:
            log(f"Frame {current_frame} wird verarbeitet")
        if external_scene_data is None and IN_BLENDER is not None:
            bpy.data.scenes[0].frame_set(current_frame)

        if t == 0 or t % 5 == 0:
            BlenderKI.update_progress("Aufgabe 4: controller_durchlauf", ((1.0 / frameLength) * t))

        for index in settings.obj_info_dict:
            if index not in objEinmalAktiv:
                aktiverState[index] = "start"
                aktiverStateKopie[index] = "start"
                zeitDict[index] = {}
                einmalAktivDict[index] = {}
                objEinmalKeyIndexDict[index] = 0
                pointDictDict[index] = {}
                settings.dynamic_obj_dict[index] = {}
                objEinmalAktiv[index] = index
            
            if external_scene_data is not None and IN_BLENDER is None:
                target = next((o for o in external_scene_data["objects"] if o["name"] == index), None)
                current_obj_location = mathutils.Vector((target["location"][0], target["location"][1], target["location"][2])) 
            else:
                current_obj_location = settings.obj_info_dict[index]['obj'].location
                
            current_state = next((i for i, s in settings.obj_info_dict[index]['state_dict'].items()
                                  if s['state_name'] == aktiverState[index]), 0)
            if aktiverStateKopie[index] != aktiverState[index]:
                zeitDict[index] = {}
                einmalAktivDict[index] = {}
                objEinmalKeyIndexDict[index] = 0
                pointDictDict[index] = {}
                aktiverStateKopie[index] = aktiverState[index]

            state = settings.obj_info_dict[index]['state_dict'][current_state]
            if current_frame >= int(state['frame_pos_start']):
                execute_logic_for_object(index)
                art = state['action_art']
                

                # --------------------------------------------------
                # Bewegung: von_bis_speed und von_bis_nach_zeit (mit Hindernissen)
                # --------------------------------------------------
                if art == 'von_bis_speed' or art == 'von_bis_nach_zeit':
                    if alleAgentenEinmalAktiv == False:
                        iZahl = 0
                        for i, (index2, item) in enumerate(settings.obj_info_dict.items()):
                            for i2, (s_index, state2) in enumerate(item['state_dict'].items()):
                                if (state2['action_art'] == 'von_bis_speed' or state2['action_art'] == 'von_bis_nach_zeit') and state2['hindernisArt'] == 'agent':
                                    
                                    if external_scene_data is not None and IN_BLENDER is None:
                                        _von_obj = state2['von']
                                        _bis_obj = state2['bis']
                                        target1 = next((o for o in external_scene_data["objects"] if o["name"] == _von_obj), None)
                                        start_pos = mathutils.Vector((target1["location"][0], target1["location"][1], target1["location"][2]))
                                        target2 = next((o for o in external_scene_data["objects"] if o["name"] == _bis_obj), None)
                                        start_pos = mathutils.Vector((target2["location"][0], target2["location"][1], target2["location"][2]))
                                        objWert = None
                                    else:
                                        _von_obj = bpy.data.objects[state2['von']]
                                        _bis_obj = bpy.data.objects[state2['bis']]
                                        start_pos = _von_obj.location.copy()
                                        ziel_pos = _bis_obj.location.copy()
                                        objWert = item['obj']
                                    _zeit_aktiv = False
                                    _speed = 100.0
                                    _time = 0.0
                                    if state2['action_art'] == 'von_bis_speed':
                                        _speed = float(state2['speed'])  # cm/s
                                    else:
                                        _time = float(state2['zeit_in_sek'])
                                        _zeit_aktiv = True
                                    _look = state2['look_at']
                                    hindernis_flag = state2.get('hindernisAktiv', 'nein')
                                    
                                    agentenDict[iZahl] = {
                                        'name': index2,
                                        'obj': objWert,
                                        'von_obj': _von_obj,
                                        'bis_obj': _bis_obj,
                                        'start_pos': start_pos,
                                        'ziel_pos': ziel_pos,
                                        'speed': _speed,
                                        'look': _look,
                                        'time': _time,
                                        'zeit_aktiv': _zeit_aktiv,
                                        'hindernis_flag': hindernis_flag,
                                        'moveDictNeu': {},
                                        'rotDictNeu': {}
                                    }
                                    iZahl += 1
                        hindernisseList = BlenderKI.get_all_obstacles()
                        
                        agentenDictMitHindernissen = agentenDict.copy()
                        for i in hindernisseList:
                            agentenDictMitHindernissen[iZahl] = {
                                'name': i.name,
                                'obj': i,
                                'von_obj': None,
                                'bis_obj': None,
                                'start_pos': i.location.copy(),
                                'ziel_pos': i.location.copy(),
                                'speed': 0.0,
                                'look': False,
                                'time': 0.0,
                                'zeit_aktiv': False,
                                'hindernis_flag': 'nein',
                                'moveDictNeu': {},
                                'rotDictNeu': {}
                            }
                            iZahl += 1
                        agents = BlenderKI.get_all_agents()
                        
                        BlenderKI.compute_astar_world_path( bpy.data.objects['Ziel'], agents, hindernisseList, agentenDict, startFrame, endFrame, agentenDictMitHindernissen)
                        
                        alleAgentenEinmalAktiv = True
                   


                # --------------------------------------------------
                # Bewegung: objekt_folgen (mit Hindernissen)
                # --------------------------------------------------
                elif art == 'objekt_folgen':
                    if external_scene_data is not None and IN_BLENDER is None:
                        ziel = state['ziel_obj']
                        obj = settings.obj_info_dict[index]['name']
                    else:
                        ziel = bpy.data.objects[state['ziel_obj']]
                        obj = settings.obj_info_dict[index]['obj']
                        
                    dist_grenze = float(state['distanz']) if settings.is_number(state['distanz']) else 0.0
                    
                    # Track-To-Constraint einmalig anlegen
                    if 'track_to_added' not in einmalAktivDict[index]:
                        if not any(c.type == 'TRACK_TO' for c in obj.constraints):
                            c = obj.constraints.new(type='TRACK_TO')
                            c.target = ziel
                            c.track_axis = 'TRACK_Z'
                            c.up_axis = 'UP_Y'
                        einmalAktivDict[index]['track_to_added'] = True

                    # einfache Hindernisvermeidung: wenn zu nah an einem Hindernis, seitlich wegschieben
                    hindernisse = BlenderKI.get_obstacles_from_settings()
                    avoid_vec = Vector((0.0, 0.0, 0.0))
                    for h in hindernisse:
                        d = (obj.location - h.location).length
                        if d < dist_grenze and d > 0:
                            away = (obj.location - h.location).normalized()
                            avoid_vec += away

                    if avoid_vec.length > 0:
                        avoid_vec.normalize()
                        # kleine Ausweichbewegung
                        obj.location += avoid_vec * 0.05

                    current_obj_location = obj.location
                    write_keyframes(obj, current_frame, location=current_obj_location)
                    
                # --------------------------------------------------
                # Drehung: look_at
                # --------------------------------------------------
                elif art == 'look_at':
                    if external_scene_data is not None and IN_BLENDER is None:
                        ziel_obj = state['ziel_obj']
                        target = next((o for o in external_scene_data["objects"] if o["name"] == ziel_obj), None)
                        loc = mathutils.Vector((target["location"][0], target["location"][1], target["location"][2])) 
                    else:
                        ziel_obj = bpy.data.objects[state['ziel_obj']]
                        loc = ziel_obj.location
                    
                    point_obj_update(index, 'lookAt', loc, float(state['zeit_in_sek']))
                    if external_scene_data is not None and IN_BLENDER is None:
                        target = next((o for o in external_scene_data["objects"] if o["name"] == index), None)
                        rot = mathutils.Euler((target["rotation_euler"][0], target["rotation_euler"][1], target["rotation_euler"][2]), "XYZ") 
                    else:
                        rot = settings.obj_info_dict[index]['obj'].rotation_euler
                    write_keyframes(index, current_frame, None, rot)

                # --------------------------------------------------
                # Drehung: setze_rotation
                # --------------------------------------------------
                elif art == 'setze_rotation':
                    set_rotation_to_target(index, state['ziel_obj'])
                    if external_scene_data is not None and IN_BLENDER is None:
                        target = next((o for o in external_scene_data["objects"] if o["name"] == index), None)
                        rot = mathutils.Euler((target["rotation_euler"][0], target["rotation_euler"][1], target["rotation_euler"][2]), "XYZ") 
                    else:
                        rot = settings.obj_info_dict[index]['obj'].rotation_euler
                    write_keyframes(index, current_frame, None, rot)

                # --------------------------------------------------
                # Animation: extern
                # --------------------------------------------------
                elif art == 'animation':
                    anim_name = state['animation_name']
                    cut_von = float(state['cut_von'])
                    cut_bis = float(state['cut_bis'])
                    loop = state['loop_or_once']
                    play_animation(index, anim_name, cut_von, cut_bis, loop)

                # --------------------------------------------------
                # Kamera: camera_follow
                # --------------------------------------------------
                elif art == 'camera_follow':
                    cam_name = index
                    ziel_str = state['ziel_obj']
                    ziel_namen = [z.strip() for z in ziel_str.split(',') if z.strip()]
                    dist = float(state['distanz']) if is_number(state['distanz']) else 5.0
                    camera_follow_objects(cam_name, ziel_namen, distance=dist)

                    
    
def execute_logic_for_object(obj_name):
    """
    Führt alle konstruierten Logikblöcke eines Objekts aus.
    Jeder Block kann einen State-Wechsel auslösen.
    """
    if 'konstrukt_code' not in settings.obj_info_dict[obj_name]:
        return

    code_dict = settings.obj_info_dict[obj_name]['konstrukt_code']

    # Lokaler Kontext für exec()
    local_context = {
        'state_sensor': state_sensor,
        'distanz_sensor': distanz_sensor,
        'raycast_sensor': raycast_sensor,
        'delay_sensor': delay_sensor,
        'variable_sensor': variable_sensor,
        'settings': settings,
    }

    # bpy nur in Blender erlauben
    if IN_BLENDER and external_scene_data is None:
        local_context['bpy'] = bpy

    for index, code in code_dict.items():
        try:
            exec(code, {}, local_context)
        except Exception as e:
            print(f"Fehler im konstruktCode von {obj_name}: {e}")

# ----------------------------------------------------------
# Initialisierung
# ----------------------------------------------------------
def init_controller():
    """
    Initialisiert das System:
    - liest Objekte ein
    - lädt State-Dateien
    - führt den Controller über einen Framebereich aus
    """
    # Annahme: BlenderKI.check_prop() und BlenderKI.get_state_txt_files()
    # wurden bereits aufgerufen, wenn dieses Modul direkt genutzt wird.
    #scene = bpy.context.scene
    #startFrame = scene.frame_start
    #endFrame = scene.frame_end
    #controller_durchlauf(startFrame, endFrame)
    #apply_keyframes_from_txt()


    #set_all_fcurves_linear()
