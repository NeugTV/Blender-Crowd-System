try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

import os, sys
import math
import json
import datetime

# mathutils nur in Blender verwenden
if IN_BLENDER:
    from mathutils import Vector, Euler
    import mathutils

external_scene_data = None

import settings
import BlenderKI

TURN_DURATION = 0.25      # Sekunden für Drehung zum nächsten Wegpunkt

LOGFILE = "blenderki_debug_log.txt"


def log(msg):
    """Schreibt Debug-Informationen in eine TXT-Datei."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


def write_keyframes(obj_name, frame, pos, rot):

    # Frame-Key als String
    frame_key = str(frame)

    # Sicherstellen, dass keyframes_dict existiert
    if "keyframes_dict" not in settings.obj_info_dict[obj_name]:
        settings.obj_info_dict[obj_name]["keyframes_dict"] = {}

    # Sicherstellen, dass der Frame-Eintrag existiert
    if frame_key not in settings.obj_info_dict[obj_name]["keyframes_dict"]:
        settings.obj_info_dict[obj_name]["keyframes_dict"][frame_key] = {}

    # Position speichern
    if pos is not None:
        settings.obj_info_dict[obj_name]["keyframes_dict"][frame_key]["pos"] = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2])
        ]

    # Rotation speichern
    if rot is not None:
        settings.obj_info_dict[obj_name]["keyframes_dict"][frame_key]["rot"] = [
            float(rot[0]),
            float(rot[1]),
            float(rot[2])
        ]


def frequenz(freq):
    if external_scene_data is not None and not IN_BLENDER:
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


def _get_location_headless(name):
    """Hilfsfunktion: Position aus external_scene_data holen."""
    if external_scene_data is None:
        return (0.0, 0.0, 0.0)
    for o in external_scene_data["objects"]:
        if o["name"] == name:
            loc = o["location"]
            return (float(loc[0]), float(loc[1]), float(loc[2]))
    return (0.0, 0.0, 0.0)


def distanz_sensor(objName, zielObjName, distValue, freq=0):
    if not frequenz(freq):
        return False

    if external_scene_data is not None and not IN_BLENDER:
        p1 = _get_location_headless(objName)
        p2 = _get_location_headless(zielObjName)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    else:
        p1 = bpy.data.objects[objName].location
        p2 = bpy.data.objects[zielObjName].location
        dist = (p2 - p1).length

    return dist <= abs(distValue)


def raycast_sensor(obj, zielName, direction, distance, freq=0):
    """
    Blender: obj ist bpy-Objekt, direction ist Vector.
    Headless: obj ist Objektname (String), direction ist (x,y,z)-Tupel.
    """
    if not frequenz(freq):
        return False

    # Blender-Version
    if IN_BLENDER and external_scene_data is None:
        origin = obj.location
        result, loc, normal, index, obj2, matrix = bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph, origin, direction
        )
        return result and obj2.name == zielName and (origin - loc).length <= distance

    # Headless-Version
    origin = _get_location_headless(obj)
    target_pos = _get_location_headless(zielName)

    vec = (
        target_pos[0] - origin[0],
        target_pos[1] - origin[1],
        target_pos[2] - origin[2],
    )

    dx, dy, dz = direction
    dot = vec[0]*dx + vec[1]*dy + vec[2]*dz
    if dot <= 0:
        return False

    dist = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return dist <= distance


def delay_sensor(delayFrames, startFrame, loopOrOnce):
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
    """
    Schritt 1 (Blender): Kamera wirklich bewegen.
    Schritt 2 (Headless): Keyframes in obj_info_dict schreiben.
    """
    # Headless: nur Keyframes berechnen
    if external_scene_data is not None and not IN_BLENDER:
        # Mittelpunkt der Zielobjekte
        if not obj_list:
            return
        xs, ys, zs = [], [], []
        for name in obj_list:
            x, y, z = _get_location_headless(name)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)

        # Kamera-Position: hinter und über dem Zentrum
        cam_x = cx
        cam_y = cy - distance
        cam_z = cz + distance

        # Rotation: auf Zentrum schauen (nur Z-Achse)
        dx = cx - cam_x
        dy = cy - cam_y
        angle = math.atan2(dy, dx)
        rot = (0.0, 0.0, angle)

        frame = external_scene_data["current_frame"]
        write_keyframes(cam_name, frame, (cam_x, cam_y, cam_z), rot)
        return

    # Blender-Version
    if not IN_BLENDER:
        return

    cam = bpy.data.objects[cam_name]
    center = Vector((0, 0, 0))
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


def look_at_angle(from_pos, to_pos):
    """
    Berechnet den Winkel (in Radiant) um Z, damit das Objekt von from_pos nach to_pos schaut.
    from_pos / to_pos können Vectors oder (x,y,z)-Tupel sein.
    """
    if IN_BLENDER and isinstance(from_pos, (Vector, )):
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
    else:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

    if dx == 0 and dy == 0:
        return None
    angle = math.atan2(dy, dx)
    return angle


def animate_agent(obj, world_path, fps, start_frame, speed_cm_s, zeitAktiv, zeit_in_sek):
    """
    Blender: obj ist bpy-Objekt, world_path: [(Vector, t), ...]
    Headless: obj ist Objektname (String), world_path: [((x,y,z), t), ...]
    Rückgabe: moveDict, rotDict mit reinen Zahlen-Tupeln.
    """
    rotDict = {}
    moveDict = {}
    rotZeitDict = {}
    moveZeitDict = {}

    if len(world_path) < 2:
        return moveDict, rotDict

    # Gesamtdistanz
    total_dist = 0.0
    for i in range(1, len(world_path)):
        p0, _ = world_path[i-1]
        p1, _ = world_path[i]
        if IN_BLENDER and isinstance(p0, (Vector, )):
            seg = (p1 - p0).length
        else:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            seg = math.sqrt(dx*dx + dy*dy + dz*dz)
        total_dist += seg

    if total_dist == 0:
        return moveDict, rotDict

    total_time = total_dist / (speed_cm_s / 100.0)
    time_per_meter = total_time / total_dist

    current_frame = start_frame
    turn_end_frame_last = current_frame
    last_rot = None

    for i in range(1, len(world_path)):
        p0, t0 = world_path[i-1]
        p1, t1 = world_path[i]

        if IN_BLENDER and isinstance(p0, (Vector, )):
            segment_dist = (p1 - p0).length
        else:
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            segment_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        segment_time = segment_dist * time_per_meter
        segment_frames = max(1, int(segment_time * fps))

        # erste Rotation/Position
        if len(rotDict) == 0:
            if IN_BLENDER and isinstance(p1, (Vector, )) and IN_BLENDER:
                rotDict[current_frame] = (0.0, 0.0, look_at_angle(p0, p1))
            else:
                angle0 = look_at_angle(p0, p1)
                rotDict[current_frame] = (0.0, 0.0, angle0 if angle0 is not None else 0.0)
        if len(moveDict) == 0:
            if IN_BLENDER and isinstance(p0, (Vector, )):
                moveDict[current_frame] = (p0.x, p0.y, p0.z)
            else:
                moveDict[current_frame] = (p0[0], p0[1], p0[2])

        # Drehung zum nächsten Wegpunkt
        angle = look_at_angle(p0, p1)
        turn_frames = max(1, int(round(TURN_DURATION * fps)))
        turn_end_frame = current_frame + turn_frames

        rot = (0.0, 0.0, angle if angle is not None else 0.0)

        if last_rot is not None:
            rotDict[turn_end_frame_last] = last_rot
            last_rot = rot
            turn_end_frame_last = current_frame
        else:
            last_rot = rot
            turn_end_frame_last = current_frame

        rotDict[turn_end_frame] = rot

        # Position am Ende des Segments
        if IN_BLENDER and isinstance(p1, (Vector, )):
            moveDict[current_frame + segment_frames] = (p1.x, p1.y, p1.z)
        else:
            moveDict[current_frame + segment_frames] = (p1[0], p1[1], p1[2])

        current_frame += segment_frames

    if zeitAktiv:
        frameSpanne = current_frame - start_frame
        gewünschteSpanne = int(zeit_in_sek * fps)
        for frame, pos in moveDict.items():
            new_frame = start_frame + int(((frame - start_frame) / max(1, frameSpanne)) * gewünschteSpanne)
            moveZeitDict[new_frame] = pos
        for frame, rot in rotDict.items():
            new_frame = start_frame + int(((frame - start_frame) / max(1, frameSpanne)) * gewünschteSpanne)
            rotZeitDict[new_frame] = rot
        return moveZeitDict, rotZeitDict
    else:
        return moveDict, rotDict


def objekt_folgen_vorher(obj_name, pointDict, start_frame, start_loc, ziel_loc, zeit_in_sek, konstant_speed=False):
    fps = settings.get_fps()
    total_frames = int(zeit_in_sek * fps)
    result = {}
    if len(pointDict) <= 1:
        if len(pointDict) == 1:
            result[start_frame] = {'wegpunkt': pointDict[0]}
        return result
    frames_per_point = max(1, int(total_frames / max(1, len(pointDict) - 1)))
    current_frame = start_frame
    for i in range(len(pointDict)):
        result[current_frame] = {'wegpunkt': pointDict[i]}
        current_frame += frames_per_point
    return result


def calculate_speed_path(start, ziel, speed_cm_s, start_frame):
    """
    Blender-Funktion (Vector-basiert). In Schritt 2 nicht verwendet.
    """
    if not IN_BLENDER:
        return {}
    fps = settings.get_fps()
    dist = (ziel - start).length
    if dist == 0:
        return {}
    total_frames = int((dist / speed_cm_s) * fps)
    result = {}
    for i in range(total_frames + 1):
        t = i / total_frames
        pos = start.lerp(ziel, t)
        result[start_frame + i] = {'wegpunkt': pos}
    return result


def point_obj_update(obj_name, modus, ziel_vec, zeit_in_sek):
    """
    Blender-Funktion. In Schritt 2 berechnen wir Rotation direkt in controller_durchlauf.
    """
    if external_scene_data is not None and not IN_BLENDER:
        return

    obj = settings.obj_info_dict[obj_name]['obj']
    loc = obj.location

    if modus in ['lookAt', 'startTrack']:
        direction = ziel_vec - loc
        obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()


def set_rotation_to_target(obj_name, ziel_name):
    """
    Blender-Funktion. In Schritt 2 berechnen wir Rotation direkt in controller_durchlauf.
    """
    if external_scene_data is not None and not IN_BLENDER:
        return

    obj = bpy.data.objects[obj_name]
    ziel = bpy.data.objects[ziel_name]
    direction = ziel.location - obj.location
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()


def play_animation(obj_name, anim_name, cut_von, cut_bis, loop_or_once):
    # Headless: Animationen existieren nicht
    if external_scene_data is not None and not IN_BLENDER:
        return

    if not IN_BLENDER:
        return

    obj = bpy.data.objects[obj_name]
    if obj.animation_data is None:
        obj.animation_data_create()
    obj.animation_data.action = bpy.data.actions.get(anim_name)


def set_all_fcurves_linear():
    if not IN_BLENDER:
        return
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


def controller_durchlauf(startFrame, endFrame):
    """
    Schritt 2: läuft nur im Headless-Modus (external_scene_data != None).
    In Blender kannst du ihn zum Debuggen auch laufen lassen, dann werden
    Keyframes direkt in obj_info_dict geschrieben.
    """
    log(f"Starte controller_durchlauf: start={startFrame}, end={endFrame}")
    aktiverStateKopie = {}
    objEinmalAktiv = {}
    zeitDict = {}
    einmalAktivDict = {}
    pointDictDict = {}
    objEinmalKeyIndexDict = {}
    alleAgentenEinmalAktiv = False
    agentenDict = {}
    frameLength = endFrame - startFrame

    # Keyframe-Struktur vorbereiten
    BlenderKI.update_progress("Aufgabe 1: Vorbereitung der Keyframe-Struktur in settings.obj_info_dict", 0.0)
    for index in settings.obj_info_dict:
        BlenderKI.update_progress("Aufgabe 1: Vorbereitung der Keyframe-Struktur in settings.obj_info_dict", (1.0 / len(settings.obj_info_dict)) * list(settings.obj_info_dict.keys()).index(index))
        if "keyframes_dict" not in settings.obj_info_dict[index]:
            settings.obj_info_dict[index]["keyframes_dict"] = {}
    BlenderKI.update_progress("Aufgabe 1: Vorbereitung der Keyframe-Struktur in settings.obj_info_dict", 1.0)
    
    BlenderKI.update_progress("Aufgabe 2: Hauptschleife über die Frames", 0.0)

    for t in range(frameLength):
        BlenderKI.update_progress("Aufgabe 2: Hauptschleife über die Frames", ((1.0 / len(range(frameLength))) * t))
        current_frame = startFrame + t

        if t % 10 == 0:
            log(f"Frame {current_frame} wird verarbeitet")

        if external_scene_data is None and IN_BLENDER:
            bpy.data.scenes[0].frame_set(current_frame)

        BlenderKI.update_progress("Aufgabe 3: Logik für alle Objekte ausführen", 0.0)
        for index in settings.obj_info_dict:
            BlenderKI.update_progress("Aufgabe 3: Logik für alle Objekte ausführen", (1.0 / len(settings.obj_info_dict)) * list(settings.obj_info_dict.keys()).index(index))
            if index not in objEinmalAktiv:
                settings.aktiverState[index] = "0"
                aktiverStateKopie[index] = "0"
                zeitDict[index] = {}
                einmalAktivDict[index] = {}
                objEinmalKeyIndexDict[index] = 0
                pointDictDict[index] = {}
                objEinmalAktiv[index] = index

            if external_scene_data is not None and not IN_BLENDER:
                current_obj_location = _get_location_headless(index)
            else:
                current_obj_location = settings.obj_info_dict[index]['obj'].location

            current_state = settings.aktiverState[index]
            # Auto-Start: Wenn ein Objekt keinen Konstrukt-Code hat,
            # wird State 0 automatisch aktiviert.
            if 'konstrukt_code' not in settings.obj_info_dict[index] or \
            not settings.obj_info_dict[index]['konstrukt_code']:
                settings.aktiverState[index] = "0"
                aktiverStateKopie[index] = "0"
                
            if aktiverStateKopie[index] != settings.aktiverState[index]:
                zeitDict[index] = {}
                einmalAktivDict[index] = {}
                objEinmalKeyIndexDict[index] = 0
                pointDictDict[index] = {}
                aktiverStateKopie[index] = settings.aktiverState[index]
            
            print(f"current_state: {current_state} objekt: {index} aktiverStateDict: {str(len(settings.obj_info_dict[index]['state_dict']))}")
            state = settings.obj_info_dict[index]['state_dict'].get(current_state)
            if state is None:
                print(f"Warnung: Kein State gefunden für Objekt {index} mit current_state {current_state}")
            if current_frame >= int(state['frame_pos_start']):
                execute_logic_for_object(index)
                art = state['action_art']
                # --------------------------------------------------
                # Bewegung: von_bis_speed und von_bis_nach_zeit (mit Hindernissen)
                # --------------------------------------------------
                if art in ('von_bis_speed', 'von_bis_nach_zeit'):
                    if not alleAgentenEinmalAktiv:
                        iZahl = 0
                        for index2, item in settings.obj_info_dict.items():
                            for s_index, state2 in item['state_dict'].items():
                                if (state2['action_art'] in ('von_bis_speed', 'von_bis_nach_zeit')
                                        and state2['hindernisArt'] == 'agent'):

                                    if external_scene_data is not None and not IN_BLENDER:
                                        _von_obj = state2['von']
                                        _bis_obj = state2['bis']
                                        start_pos = _get_location_headless(_von_obj)
                                        ziel_pos = _get_location_headless(_bis_obj)
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

                        # Hindernisse nur in Blender-Variante als Objekte
                        if IN_BLENDER and external_scene_data is None:
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

                        if external_scene_data is not None and not IN_BLENDER:
                            agents = BlenderKI.get_all_agents()
                            BlenderKI.compute_astar_world_path(
                                state['ziel_obj'],
                                agents,
                                hindernisseList,
                                agentenDict,
                                startFrame,
                                endFrame,
                                agentenDictMitHindernissen,
                                current_state
                            )
                        else:
                            agents = [settings.obj_info_dict[a]['obj'] for a in BlenderKI.get_all_agents()]
                            BlenderKI.compute_astar_world_path(
                                bpy.data.objects[state['ziel_obj']],
                                agents,
                                hindernisseList,
                                agentenDict,
                                startFrame,
                                endFrame,
                                agentenDictMitHindernissen,
                                current_state
                            )

                        alleAgentenEinmalAktiv = True

                # --------------------------------------------------
                # Bewegung: objekt_folgen (mit Hindernissen)
                # --------------------------------------------------
                elif art == 'objekt_folgen':
                    if external_scene_data is not None and not IN_BLENDER:
                        ziel_name = state['ziel_obj']
                        obj_name = index
                        obj_pos = _get_location_headless(obj_name)
                        ziel_pos = _get_location_headless(ziel_name)
                        dist_grenze = float(state['distanz']) if settings.is_number(state['distanz']) else 0.0

                        # einfache Hindernisvermeidung: hier nur Platzhalter
                        # direktes Folgen:
                        write_keyframes(obj_name, current_frame, obj_pos, None)
                    else:
                        ziel = bpy.data.objects[state['ziel_obj']]
                        obj = settings.obj_info_dict[index]['obj']
                        dist_grenze = float(state['distanz']) if settings.is_number(state['distanz']) else 0.0

                        if 'track_to_added' not in einmalAktivDict[index]:
                            if not any(c.type == 'TRACK_TO' for c in obj.constraints):
                                c = obj.constraints.new(type='TRACK_TO')
                                c.target = ziel
                                c.track_axis = 'TRACK_Z'
                                c.up_axis = 'UP_Y'
                            einmalAktivDict[index]['track_to_added'] = True

                        hindernisse = BlenderKI.get_obstacles_from_settings()
                        avoid_vec = Vector((0.0, 0.0, 0.0))
                        for h in hindernisse:
                            d = (obj.location - h.location).length
                            if d < dist_grenze and d > 0:
                                away = (obj.location - h.location).normalized()
                                avoid_vec += away

                        if avoid_vec.length > 0:
                            avoid_vec.normalize()
                            obj.location += avoid_vec * 0.05

                        current_obj_location = obj.location
                        write_keyframes(index, current_frame, (current_obj_location.x,
                                                               current_obj_location.y,
                                                               current_obj_location.z), None)

                # --------------------------------------------------
                # Drehung: look_at
                # --------------------------------------------------
                elif art == 'look_at':
                    ziel_name = state['ziel_obj']
                    if external_scene_data is not None and not IN_BLENDER:
                        obj_pos = _get_location_headless(index)
                        ziel_pos = _get_location_headless(ziel_name)
                        angle = look_at_angle(obj_pos, ziel_pos)
                        rot = (0.0, 0.0, angle if angle is not None else 0.0)
                        write_keyframes(index, current_frame, None, rot)
                    else:
                        ziel_obj = bpy.data.objects[ziel_name]
                        loc = ziel_obj.location
                        point_obj_update(index, 'lookAt', loc, float(state['zeit_in_sek']))
                        rot = settings.obj_info_dict[index]['obj'].rotation_euler
                        write_keyframes(index, current_frame, None,
                                        (rot.x, rot.y, rot.z))

                # --------------------------------------------------
                # Drehung: setze_rotation
                # --------------------------------------------------
                elif art == 'setze_rotation':
                    ziel_name = state['ziel_obj']
                    if external_scene_data is not None and not IN_BLENDER:
                        obj_pos = _get_location_headless(index)
                        ziel_pos = _get_location_headless(ziel_name)
                        angle = look_at_angle(obj_pos, ziel_pos)
                        rot = (0.0, 0.0, angle if angle is not None else 0.0)
                        write_keyframes(index, current_frame, None, rot)
                    else:
                        set_rotation_to_target(index, ziel_name)
                        rot = settings.obj_info_dict[index]['obj'].rotation_euler
                        write_keyframes(index, current_frame, None,
                                        (rot.x, rot.y, rot.z))

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
                    
        BlenderKI.update_progress("Aufgabe 3: Logik für alle Objekte ausgeführt", 1.0)
    BlenderKI.update_progress("Aufgabe 2: Hauptschleife über die Frames", 1.0)


def execute_logic_for_object(obj_name):
    """
    Führt alle konstruierten Logikblöcke eines Objekts aus.
    Jeder Block kann einen State-Wechsel auslösen.
    """
    if 'konstrukt_code' not in settings.obj_info_dict[obj_name]:
        return

    code_dict = settings.obj_info_dict[obj_name]['konstrukt_code']

    local_context = {
        'state_sensor': state_sensor,
        'distanz_sensor': distanz_sensor,
        'raycast_sensor': raycast_sensor,
        'delay_sensor': delay_sensor,
        'variable_sensor': variable_sensor,
        'settings': settings,
    }

    if IN_BLENDER and external_scene_data is None:
        local_context['bpy'] = bpy

    for index, code in code_dict.items():
        try:
            exec(code, {}, local_context)
        except Exception as e:
            print(f"Fehler im konstruktCode von {obj_name}: {e}")


def apply_keyframes_from_obj_info_dict():
    """
    Schritt 3: Liest alle Keyframes aus settings.obj_info_dict und
    schreibt sie als echte Blender-Keyframes in die Szene.
    """
    log("apply_keyframes_from_obj_info_dict: Starte das Anwenden der Keyframes in Blender.")
    if not IN_BLENDER:
        log("apply_keyframes_from_obj_info_dict: Nicht in Blender, keine Keyframes gesetzt.")
        return

    for obj_name, sub in settings.obj_info_dict.items():

        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue

        keyframes = sub.get("keyframes_dict", {})

        for frame_str, data in keyframes.items():

            # Frame sicher in int umwandeln
            try:
                frame = int(frame_str)
            except:
                log(f"Ungültiger Frame-Wert für {obj_name}: {frame_str}")
                continue

            # Position extrahieren
            pos = data.get("pos")
            rot = data.get("rot")

            if pos is not None:
                try:
                    log(f"Setze Position-Keyframe für {obj_name} bei Frame {frame}: {pos}")
                    obj.location = Vector((
                        float(pos[0]),
                        float(pos[1]),
                        float(pos[2])
                    ))
                    obj.keyframe_insert(data_path="location", frame=frame)
                except Exception as e:
                    print("Fehler bei Position-Keyframe:", obj_name, frame, e)

            if rot is not None:
                try:
                    obj.rotation_euler = Euler((
                        float(rot[0]),
                        float(rot[1]),
                        float(rot[2])
                    ), "XYZ")
                    obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                except Exception as e:
                    print("Fehler bei Rotations-Keyframe:", obj_name, frame, e)
