# BlenderKI.py
# Zentrale KI-Funktionen: Objekt-Initialisierung, State-Handling, A*-Navigation

import bpy
import math
from mathutils import Vector, Euler, Quaternion
import mathutils
import bmesh
import os.path
import time, sys, os
import heapq

dir = os.path.dirname(bpy.data.filepath)
if dir not in sys.path:
    sys.path.append(dir)

import settings
import controller

# ----------------------------------------------------------
# Hilfsfunktion: Fortschrittsanzeige in der Konsole
# ----------------------------------------------------------
def update_progress(job_title, progress):
    length = 20
    block = int(round(length * progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title,
                                     "#" * block + "-" * (length - block),
                                     round(progress * 100, 2))
    if progress >= 1:
        msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

# ----------------------------------------------------------
# Objekte einlesen und Grundstruktur in settings.obj_info_dict anlegen
# ----------------------------------------------------------
def check_prop():
    """
    Durchläuft alle Objekte in der Szene und legt für Objekte mit
    obj_eigenschaft == 'aktiv' einen Eintrag in settings.obj_info_dict an.
    """
    objLength = len(bpy.data.objects)
    for i, ob in enumerate(bpy.data.objects):
        update_progress("Aufgabe 1: check_prop", ((1.0 / objLength) * i))
        obj_eigenschaft = ''
        obj_eigenschaft2 = ''
        obj_eigenschaft3 = ''
        if len(ob.keys()) > 0:
            for K in ob.keys():
                if K == 'obj_eigenschaft':
                    obj_eigenschaft = ob[K]
                if K == 'hindernisAktiv':
                    obj_eigenschaft2 = ob[K]
                if K == 'hindernisArt':
                    obj_eigenschaft3 = ob[K]

        if obj_eigenschaft == 'aktiv':
            keyframes_dict = {}
            state_dict_Einzelnd = {
                'state_name': 'start',
                'frame_pos_start': '0',
                'action_art': '',
                'animation_name': '',
                'ziel_obj': '',
                'distanz': '0',
                'verfolgen_fliehen': '0',
                'von': '',
                'bis': '',
                'zeit_in_sek': '0.0',
                'speed': '0.0',
                'cut_von': '0',
                'cut_bis': '0',
                'loop_or_once': '',
                'hindernisAktiv': 'nein',
                'hindernisArt': 'statisch',
                'zuweisung_acturator_sensor': '',
                'keyframe_modus': 'neu',
                'boden_name': '',
                'look_at': 'ja'
            }
            if obj_eigenschaft2 != '':
                state_dict_Einzelnd['hindernisAktiv'] = obj_eigenschaft2
            if obj_eigenschaft3 != '':
                state_dict_Einzelnd['hindernisArt'] = obj_eigenschaft3

            state_dict = {0: state_dict_Einzelnd}
            variablen_Dict = {}
            konstrukt_code_dict = {}

            sub_info_dict = {
                'obj': ob,
                'keyframes_dict': keyframes_dict,
                'state_dict': state_dict,
                'variablen': variablen_Dict,
                'konstrukt_code': konstrukt_code_dict
            }
            settings.obj_info_dict[ob.name] = sub_info_dict

    update_progress("Aufgabe 1: check_prop", 1)

# ----------------------------------------------------------
# State-TXT-Dateien einlesen / erzeugen
# ----------------------------------------------------------
def get_state_txt_files():

    objInfoDictLength = len(settings.obj_info_dict.items())
    # bekomme value von settings.obj_info_dict von state txt files.
    # wenn die txt dateien nicht vorhanden sind werden sie erstellt und mit standart werten gefüllt
    trennung1 = '***'
    trennung2 = '###'
    trennung3 = '|'
    trennung4 = ','
    trennung5 = '$'
    
    for objInfDiIndex, (objName, subInfoDict) in enumerate(settings.obj_info_dict.items()): #count anzahl der obj namen
        update_progress("Aufgabe 2: get_state_txt_files", ((1.0 / objInfoDictLength) * objInfDiIndex))
        stateFileName = objName + '_state.txt'
        stateFileInhalt = ''
        if os.path.isfile(stateFileName):
            with open(stateFileName, 'r') as file:
                stateFileInhalt = file.read().replace('\n', ' ')
            
            for subInfDiIndex, (ind, ite) in enumerate(subInfoDict.items()): # count 5
                if ind == 'state_dict':
                    i = 0
                    for i in range(len(stateFileInhalt.split(trennung1))): # count anzahl der states
                        stateArray = stateFileInhalt.split(trennung1)[i].split(trennung2)
                        state_dict = {'state_name' : 'start', 'frame_pos_start' : '0', 'action_art' : '', 'animation_name' : '', 'ziel_obj' : '', 
                         'distanz' : '0', 'verfolgen_fliehen' : '0', 'von' : '', 'bis' : '', 'zeit_in_sek' : '0.0', 'speed' : '0.0', 'cut_von': '0', 
                         'cut_bis' : '0', 'loop_or_once' : '', 'hindernisAktiv' : 'nein', 'hindernisArt' : 'statisch', 'zuweisung_acturator_sensor' : '', 
                         'keyframe_modus' : 'neu', 'boden_name' : '', 'look_at' : 'ja'}
                        i2 = 0
                        for staDiIndex, (index, item) in enumerate(state_dict.items()): # count anzahl der werte
                            if i2 <= len(stateArray) - 1:
                                state_dict[index] = stateArray[i2].split(trennung3)[1]
                            i2 += 1
                        settings.obj_info_dict[objName]['state_dict'][i] = state_dict
                elif ind == 'obj':
                    settings.obj_info_dict[objName][ind] = bpy.data.objects[objName]
                #elif index == 'keyframes_dict':
        else:
            for subInfDiIndex, (index, item) in enumerate(subInfoDict.items()): # count 5
                if index == 'state_dict':
                    i = 0
                    stateFileInhaltEinzelnd = ''
                    einStandardStateDict = item[0]
                    for eStanStaDiIndex, (index2, item2) in enumerate(einStandardStateDict.items()): # count anzahl der werte
                        stateFileInhaltEinzelnd2 = item2
                        if i == len(einStandardStateDict) - 1:
                            stateFileInhaltEinzelnd += index2 + trennung3 + stateFileInhaltEinzelnd2
                        else:
                            stateFileInhaltEinzelnd += index2 + trennung3 + stateFileInhaltEinzelnd2 + trennung2
                        i += 1
                    stateFileInhalt = stateFileInhaltEinzelnd
            dateiName = objName + '_state.txt'
            with open(dateiName, 'w') as file:
                file.write(stateFileInhalt)
    update_progress("Aufgabe 2: get_state_txt_files", 1)
    
# ----------------------------------------------------------
# A*-Navigation (Grid-basiert, mit Hindernissen)
# ----------------------------------------------------------


# Diese Konstanten kannst du auch nach settings.py verschieben, wenn du magst
CELL_SIZE = 0.75          # Rastergröße
GRID_MARGIN = 2.0         # Rand um Szenenbereich
OBSTACLE_BUFFER = 0.3     # Mindestabstand zu Hindernissen (Meter)
GOAL_SPACING = 1.2         # Abstand zwischen Agenten am Ziel (in Metern)

TURN_DURATION = 0.25      # Sekunden für Drehung zum nächsten Wegpunkt

OBSTACLE_PROP = "hindernisAktiv"
OBSTACLE_VALUE = "ja"


def get_obstacles_from_settings():
    """Alle Objekte, die in den States als Hindernis markiert sind."""
    obs = []
    for name, sub in settings.obj_info_dict.items():
        state0 = sub['state_dict'].get(0, {})
        if state0.get('hindernisAktiv') == 'ja':
            obj = sub['obj']
            if obj.type == 'MESH':
                obs.append(obj)
    return obs


def world_bbox_corners(obj):
    mat = obj.matrix_world
    return [mat @ mathutils.Vector(corner) for corner in obj.bound_box]


def bbox_min_max(obj):
    corners = world_bbox_corners(obj)
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def point_in_bbox_2d_with_buffer(x, y, obj, buffer):
    (minx, maxx), (miny, maxy), _ = bbox_min_max(obj)
    return (minx - buffer <= x <= maxx + buffer) and (miny - buffer <= y <= maxy + buffer)


def build_grid_bounds(agents, goal_positions, obstacles):
    xs, ys = [], []
    for ag in agents:
        xs.append(ag.location.x)
        ys.append(ag.location.y)
    for gp in goal_positions:
        xs.append(gp.x)
        ys.append(gp.y)
    for ob in obstacles:
        (minx, maxx), (miny, maxy), _ = bbox_min_max(ob)
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])

    min_x = min(xs) - GRID_MARGIN
    max_x = max(xs) + GRID_MARGIN
    min_y = min(ys) - GRID_MARGIN
    max_y = max(ys) + GRID_MARGIN

    return min_x, max_x, min_y, max_y


def world_to_grid(x, y, min_x, min_y):
    gx = int((x - min_x) / CELL_SIZE)
    gy = int((y - min_y) / CELL_SIZE)
    return gx, gy


def grid_to_world(gx, gy, min_x, min_y, z):
    x = min_x + gx * CELL_SIZE + CELL_SIZE * 0.5
    y = min_y + gy * CELL_SIZE + CELL_SIZE * 0.5
    return Vector((x, y, z))


def is_cell_blocked(gx, gy, min_x, min_y, obstacles, buffer=0.0):
    x = min_x + gx * CELL_SIZE + CELL_SIZE * 0.5
    y = min_y + gy * CELL_SIZE + CELL_SIZE * 0.5
    for ob in obstacles:
        if point_in_bbox_2d_with_buffer(x, y, ob, buffer):
            return True
    return False


def neighbors(gx, gy):
    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    for dx, dy in dirs:
        yield gx + dx, gy + dy


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)




def a_star(self_agent, all_agents, agent_width, start, goal, min_x, min_y, max_x, max_y, obstacles, reserved=None, buffer=0.0):
    """
    A* auf 2D‑Grid mit Zeitdimension.
    reserved: dict[(gx,gy,time_step)] = True für bereits belegte Zellen (andere Agenten).
    buffer: Sicherheitsabstand zu Hindernissen.
    """
    if reserved is None:
        reserved = {}

    start_g = start
    goal_g = goal

    open_set = []
    heapq.heappush(open_set, (0, start_g, 0))  # (f, (gx,gy), time_step)
    came_from = {}
    g_score = { (start_g, 0): 0 }

    max_gx = int((max_x - min_x) / CELL_SIZE) + 1
    max_gy = int((max_y - min_y) / CELL_SIZE) + 1

    while open_set:
        _, current, t = heapq.heappop(open_set)

        if current == goal_g:
            # Pfad rekonstruieren (mit Zeit)
            path = []
            key = (current, t)
            while key in came_from:
                (cell, time) = key
                path.append((cell, time))
                key = came_from[key]
            (cell, time) = key
            path.append((cell, time))
            path.reverse()
            return path

        for nx, ny in neighbors(*current):
            if nx < 0 or ny < 0 or nx >= max_gx or ny >= max_gy:
                continue
            # Weltposition der Zielzelle
            wx, wy, wz = grid_to_world(nx, ny, min_x, min_y, 0)
            world_pos = mathutils.Vector((wx, wy, self_agent.location.z))

            # Raycast-Kollisionsprüfung mit anderen Agenten
            if agent_collision_raycast(world_pos, all_agents, self_agent, distance=agent_width):
                # Agent bleibt stehen
                nx2, ny2 = current
                nt2 = t + 1
                if not reserved.get((nx2, ny2, nt2), False):
                    neighbor_cell = (nx2, ny2)
                    tentative_g = g_score[(current, t)] + 1
                    key_neighbor = (neighbor_cell, nt2)
                    if key_neighbor not in g_score or tentative_g < g_score[key_neighbor]:
                        g_score[key_neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor_cell, goal_g)
                        heapq.heappush(open_set, (f, neighbor_cell, nt2))
                        came_from[key_neighbor] = (current, t)
                continue
            if is_cell_blocked(nx, ny, min_x, min_y, obstacles, buffer=buffer):
                continue

            nt = t + 1

            # Kollision mit anderen Agenten vermeiden
            if reserved.get((nx, ny, nt), False):
                # Möglichkeit: stehen bleiben (in aktueller Zelle bleiben)
                nx2, ny2 = current
                nt2 = t + 1
                if not reserved.get((nx2, ny2, nt2), False):
                    neighbor_cell = (nx2, ny2)
                    tentative_g = g_score[(current, t)] + 1
                    key_neighbor = (neighbor_cell, nt2)
                    if key_neighbor not in g_score or tentative_g < g_score[key_neighbor]:
                        g_score[key_neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor_cell, goal_g)
                        heapq.heappush(open_set, (f, neighbor_cell, nt2))
                        came_from[key_neighbor] = (current, t)
                continue

            neighbor_cell = (nx, ny)
            tentative_g = g_score[(current, t)] + 1
            key_neighbor = (neighbor_cell, nt)
            if key_neighbor not in g_score or tentative_g < g_score[key_neighbor]:
                g_score[key_neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor_cell, goal_g)
                heapq.heappush(open_set, (f, neighbor_cell, nt))
                came_from[key_neighbor] = (current, t)

    return None

def expand_move_dict(moveDict):
    # sortierte Frames
    frames = sorted(moveDict.keys())
    newDict = {}

    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end = frames[i+1]

        v_start = moveDict[f_start]
        v_end = moveDict[f_end]

        # Startwert übernehmen
        newDict[f_start] = v_start

        # alle Zwischenframes interpolieren
        frame_range = f_end - f_start
        for f in range(f_start + 1, f_end):
            t = (f - f_start) / frame_range
            v = v_start.lerp(v_end, t)
            newDict[f] = v

    # letzten Frame hinzufügen
    last_frame = frames[-1]
    newDict[last_frame] = moveDict[last_frame]

    return newDict


def expand_rotation_dict(rotDict):
    frames = sorted(rotDict.keys())
    newDict = {}

    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end = frames[i+1]

        e_start = rotDict[f_start]
        e_end = rotDict[f_end]

        q_start = e_start.to_quaternion()
        q_end = e_end.to_quaternion()

        newDict[f_start] = e_start

        frame_range = f_end - f_start
        for f in range(f_start + 1, f_end):
            t = (f - f_start) / frame_range
            q = Quaternion.slerp(q_start, q_end, t)
            newDict[f] = q.to_euler()

    last_frame = frames[-1]
    newDict[last_frame] = rotDict[last_frame]

    return newDict





def grid_path_to_world(path_g, min_x, min_y, z):
    pts = []
    for (gx, gy), t in path_g:
        x, y, z0 = grid_to_world(gx, gy, min_x, min_y, z)
        pts.append((mathutils.Vector((x, y, z)), t))
    return pts

def face_direction(obj, target_pos):
    direction = (target_pos - obj.location).normalized()
    rot = direction.to_track_quat('Y', 'Z').to_euler()
    return rot


        
def get_agent_width(agent):
    (minx, maxx), (miny, maxy), _ = bbox_min_max(agent)
    return max(maxx - minx, maxy - miny)


def agent_collision_raycast(world_pos, agents, self_agent, distance=1.0):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    directions = [
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((-1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, -1, 0)),
    ]

    for ag in agents:
        if ag == self_agent:
            continue

        eval_obj = ag.evaluated_get(depsgraph)

        for d in directions:
            hit, loc, normal, face = eval_obj.ray_cast(world_pos, d)
            if hit and (loc - world_pos).length <= distance:
                return True

    return False
    
def compute_free_direction(pos, agents, self_agent, radius):
    directions = [
        Vector((1, 0, 0)),
        Vector((-1, 0, 0)),
        Vector((0, 1, 0)),
        Vector((0, -1, 0)),
        Vector((1, 1, 0)).normalized(),
        Vector((1, -1, 0)).normalized(),
        Vector((-1, 1, 0)).normalized(),
        Vector((-1, -1, 0)).normalized(),
    ]

    best_dir = Vector((0, 0, 0))
    best_dist = -1.0

    for d in directions:
        test_pos = pos + d * radius
        min_dist = 9999.0

        for ag in agents:
            if ag == self_agent:
                continue
            dist = (test_pos - ag.location).length
            if dist < min_dist:
                min_dist = dist

        if min_dist > best_dist:
            best_dist = min_dist
            best_dir = d

    return best_dir


def ray_hits_object(origin, direction, max_dist, target_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = target_obj.evaluated_get(depsgraph)
    hit, loc, normal, face = eval_obj.ray_cast(origin, direction)
    if not hit:
        return False
    return (loc - origin).length <= max_dist

# Richtung mit dem meisten freien Raum bestimmen
def compute_free_direction_geometric(pos, agents, self_agent, radius):
    directions = [
        Vector((1, 0, 0)),
        Vector((-1, 0, 0)),
        Vector((0, 1, 0)),
        Vector((0, -1, 0)),
        Vector((1, 1, 0)).normalized(),
        Vector((1, -1, 0)).normalized(),
        Vector((-1, 1, 0)).normalized(),
        Vector((-1, -1, 0)).normalized(),
    ]

    best_dir = Vector((0, 0, 0))
    best_dist = -1.0

    for d in directions:
        test_pos = pos + d * radius
        min_dist = 9999.0

        for ag in agents:
            if ag == self_agent:
                continue
            dist = (test_pos - ag.location).length
            if dist < min_dist:
                min_dist = dist

        if min_dist > best_dist:
            best_dist = min_dist
            best_dir = d

    return best_dir

def travel_time_in_frames(oldLocation, newLocation, speed_cm_per_sec):
    # oldLocation / newLocation können Tupel, Listen oder Vector sein
    old_loc = Vector(oldLocation)
    new_loc = Vector(newLocation)

    distance_cm = (new_loc - old_loc).length * 100.0  # Blender arbeitet in Metern → Umrechnung in cm
    fps = bpy.context.scene.render.fps

    time_seconds = distance_cm / speed_cm_per_sec
    frames = time_seconds * fps

    return frames

def is_point_inside_mesh(point, obj):
    """
    Überprüft, ob ein Punkt (Global Space) innerhalb eines Mesh-Objekts liegt.
    
    :param point: mathutils.Vector (Position des Punktes in World Coordinates)
    :param obj: bpy.types.Object (Das Mesh-Objekt)
    :return: bool (True, wenn innen)
    """
    # 1. Punkt in lokalen Raum des Objekts transformieren
    # Wichtig, falls das Objekt skaliert, gedreht oder verschoben ist
    mat_world_to_local = obj.matrix_world.inverted()
    local_point = mat_world_to_local @ point
    
    # 2. Raycast vorbereiten
    # Wir schießen einen Strahl in eine beliebige Richtung (z.B. +X)
    direction = mathutils.Vector((1.0, 0.0, 0.0))
    
    # 3. Schnittpunkte zählen
    count = 0
    # ray_cast benötigt eine Szene
    scene = bpy.context.scene
    
    # max_iterations verhindert Endlosschleifen bei komplexen Geometrien
    max_iterations = 100 
    
    curr_origin = local_point
    
    for _ in range(max_iterations):
        # Raycast in lokalem Raum
        hit, location, normal, index = obj.ray_cast(curr_origin, direction)
        
        if not hit:
            break
            
        count += 1
        # Neuen Ursprung leicht nach dem Treffer setzen, um nicht im Mesh stecken zu bleiben
        curr_origin = location + (direction * 0.0001)
        
    # Wenn die Anzahl der Treffer ungerade ist, ist der Punkt innen
    return count % 2 == 1

def point_in_mesh_schleife(point, objectList):
    result = False
    for obj in objectList:
        if is_point_inside_mesh(point, obj):
            result = True
            break
    return result




def sort_agents_by_last_waypoint_time(all_paths_world, agents):
    """
    Sortiert Agenten nach der Zeit t ihres letzten Wegpunkts.
    Kleinste t = höchste Priorität.
    """
    def last_t(agent):
        path = all_paths_world.get(agent.name, [])
        if not path:
            return float('inf')
        # letzter Eintrag: (pos, t)
        return path[-1][1]
    
    return sorted(agents, key=last_t)

def distribute_goal_positions(global_target, agents, spacing=1.0):
    """
    Verteilt Agenten in einem kompakten 2D‑Raster hinter dem Ziel.
    Reihenfolge der agents bestimmt die Reihenfolge im Raster.
    """
    base = global_target.location
    count = len(agents)

    # Rastergröße bestimmen (quadratisch)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    positions = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= count:
                break
            offset = Vector((
                (c - cols/2) * spacing,
                -(r * spacing),
                0.0
            ))
            positions.append(base + offset)
            idx += 1

    return positions

def compute_astar_world_path(ziel_obj, agents, obstaclesList, agentenDict, start_frame, end_frame, agentenDictMitHindernissen):
    scene = bpy.context.scene
    fps = scene.render.fps
    base_frame = scene.frame_current

    global_target = ziel_obj
    obstacles = obstaclesList

    if not agents:
        controller.log("Keine Agenten gefunden.")
        return
            
    if not global_target:
        controller.log("Globales Zielobjekt nicht gefunden.")
        return
    



    max_width = max(get_agent_width(a) for a in agents)
    GOAL_SPACING = max_width + 0.5

    # ----------------------------------------------------------
    # PHASE 1: Erste Pfade berechnen (unsortiert)
    # ----------------------------------------------------------
    goal_positions_initial = distribute_goal_positions(global_target, agents, spacing=GOAL_SPACING)
    min_x, max_x, min_y, max_y = build_grid_bounds(agents, goal_positions_initial, obstacles)

    reserved = {}
    all_paths_world = {}

    for agent, goal_pos in zip(agents, goal_positions_initial):
        start_g = world_to_grid(agent.location.x, agent.location.y, min_x, min_y)
        goal_g = world_to_grid(goal_pos.x, goal_pos.y, min_x, min_y)

        controller.log(f"Berechne Pfad für {agent.name}...")
        path_g = a_star(
            agent, agents, max_width,
            start_g, goal_g,
            min_x, min_y, max_x, max_y,
            obstacles,
            reserved=reserved,
            buffer=OBSTACLE_BUFFER
        )

        if path_g is None:
            controller.log(f"Kein Pfad für {agent.name} gefunden.")
            continue

        for (gx, gy), t in path_g:
            reserved[(gx, gy, t)] = True

        world_path = grid_path_to_world(path_g, min_x, min_y, agent.location.z)
        all_paths_world[agent.name] = world_path

    # ----------------------------------------------------------
    # PHASE 2: Agenten nach letzter t sortieren
    # ----------------------------------------------------------
    agents_sorted = sort_agents_by_last_waypoint_time(all_paths_world, agents)

    # ----------------------------------------------------------
    # PHASE 3: Neue Zielpositionen erzeugen (Raster)
    # ----------------------------------------------------------
    goal_positions_final = distribute_goal_positions(global_target, agents_sorted, spacing=GOAL_SPACING)

    # ----------------------------------------------------------
    # PHASE 4: A* erneut ausführen – diesmal mit finalen Zielpunkten
    # ----------------------------------------------------------
    reserved = {}
    all_paths_world_final = {}

    for agent, goal_pos in zip(agents_sorted, goal_positions_final):
        start_g = world_to_grid(agent.location.x, agent.location.y, min_x, min_y)
        goal_g = world_to_grid(goal_pos.x, goal_pos.y, min_x, min_y)

        controller.log(f"Berechne FINALEN Pfad für {agent.name}...")
        path_g = a_star(
            agent, agents_sorted, max_width,
            start_g, goal_g,
            min_x, min_y, max_x, max_y,
            obstacles,
            reserved=reserved,
            buffer=OBSTACLE_BUFFER
        )

        if path_g is None:
            controller.log(f"Kein finaler Pfad für {agent.name} gefunden.")
            continue

        for (gx, gy), t in path_g:
            reserved[(gx, gy, t)] = True

        world_path = grid_path_to_world(path_g, min_x, min_y, agent.location.z)
        all_paths_world_final[agent.name] = world_path

        agent["navigationsPfad"] = [
            (float(p.x), float(p.y), float(p.z), int(t)) for p, t in world_path
        ]
        

    # ----------------------------------------------------------
    # PHASE 5: moveDict und rotDict werden in agentenDict gespeichert (mit expand-Funktion)
    # ----------------------------------------------------------
    for i, (index2, item2) in enumerate(agentenDict.items()):
        agent = item2['obj']
        world_path = all_paths_world_final.get(agent.name)
        if not world_path:
            continue

        moveDict, rotDict = controller.animate_agent(
            agent,
            world_path,
            fps,
            base_frame,
            item2['speed'],
            item2['zeit_aktiv'],
            item2['time']
        )
        agentenDict[index2]['moveDictNeu'] = expand_move_dict(moveDict)
        agentenDict[index2]['rotDictNeu'] = expand_rotation_dict(rotDict)
    
    # ----------------------------------------------------------
    # PHASE 6: Kollisionsvermeidung durch "Wegschubsen" implementieren
    # ----------------------------------------------------------
    
    # 3 Wiederholungen
    for _ in range(3):
        # Alle Frames durchgehen
        for frame in range(start_frame, end_frame + 1):
            # Alle Agenten prüfen
            for index2, item2 in agentenDict.items():
                
                if frame not in item2['moveDictNeu']:
                    continue

                pos = item2['moveDictNeu'][frame]
                self_obj = item2['obj']
                self_name = self_obj.name
                agent_width = get_agent_width(self_obj)
                min_dist = agent_width * 0.6
                ray_dist = 0.5

                # 8 Strahlen (2 pro Seite)
                ray_dirs = []
                base_dirs = [
                    Vector((1, 0, 0)),
                    Vector((-1, 0, 0)),
                    Vector((0, 1, 0)),
                    Vector((0, -1, 0)),
                    Vector((1, 1, 0)).normalized(),
                    Vector((1, -1, 0)).normalized(),
                    Vector((-1, 1, 0)).normalized(),
                    Vector((-1, -1, 0)).normalized(),
                ]

                for d in base_dirs:
                    ray_dirs.append(d)
                    ray_dirs.append(Vector((d.x, d.y, 0.1)))
                    ray_dirs.append(Vector((d.x, d.y, -0.1)))

                collision_detected = False

                # Alle anderen Objekte prüfen
                for other_index, other_val in agentenDictMitHindernissen.items():

                    # Hindernisse haben keine moveDictNeu → überspringen
                    if 'moveDictNeu' not in other_val:
                        continue

                    other_obj = other_val['obj']
                    if other_obj.name == self_name:
                        continue
                    
                    if frame not in other_val['moveDictNeu'] and other_val['von_obj'] is not None:
                        continue
                    
                    # Position des anderen Objekts für Distanzcheck
                    if frame in other_val['moveDictNeu']:
                        other_pos = other_val['moveDictNeu'][frame]
                    else:
                        other_pos = other_obj.location

                    # Distanzcheck
                    if (pos - other_pos).length < min_dist:
                        collision_detected = True

                    # Geometrie‑Raycast
                    for d in ray_dirs:
                        origin = pos + d * 0.1
                        if ray_hits_object(origin, d, ray_dist, other_obj):
                            collision_detected = True
                            break

                    if collision_detected:
                        break

                # Wenn keine Kollision → weiter
                if not collision_detected:
                    continue

                
                # Richtung mit freiem Raum bestimmen
                push_dir = compute_free_direction_geometric(
                    pos,
                    [v['obj'] for v in agentenDictMitHindernissen.values()],
                    self_obj,
                    radius=min_dist
                )
                penetration = (min_dist - ray_dist)
                push_amount = min(penetration, 0.1)  # max 10 cm pro Frame
                new_pos = pos + push_dir * push_amount
                
                
                
                while point_in_mesh_schleife(new_pos, obstacles):
                    push_dir = compute_free_direction_geometric(
                        pos,
                        [v['obj'] for v in agentenDictMitHindernissen.values()],
                        self_obj,
                        radius=min_dist * 0.5
                    )
                    min_dist *= 0.5
                    if min_dist < 0.1:
                        break

                # Wegschubsen
                
                
                agentenDict[index2]['moveDictNeu'][frame] = new_pos

                # Alle folgenden Frames verschieben
                sorted_frames = sorted(item2['moveDictNeu'].keys())
                push_key = travel_time_in_frames(pos, new_pos, item2['speed'])

                for f2 in sorted_frames:
                    if f2 > frame + push_key:
                        agentenDict[index2]['moveDictNeu'][f2] = item2['moveDictNeu'][f2] + push_dir * min_dist

                # Rotation anpassen
                if frame in item2['rotDictNeu']:
                    agentenDict[index2]['rotDictNeu'][frame] = face_direction(self_obj, new_pos)
                # Nach jedem Durchlauf expandieren
            for index2, item2 in agentenDict.items():
                agentenDict[index2]['moveDictNeu'] = expand_move_dict(item2['moveDictNeu'])
                agentenDict[index2]['rotDictNeu'] = expand_rotation_dict(item2['rotDictNeu'])
                
    # ----------------------------------------------------------
    # PHASE 7: Keyframes in Blender setzen
    # ----------------------------------------------------------
    agentenDictName = ''
    for index2, item2 in agentenDict.items():
        name = item2['obj'].name
        agentenDictName += name + ', '
        for frame, pos in item2['moveDictNeu'].items():
            controller.write_keyframes(item2['obj'], frame, pos, None)

        for frame, rot in item2['rotDictNeu'].items():
            controller.write_keyframes(item2['obj'], frame, None, rot)
            
    print("Agenten mit Pfaden: " + agentenDictName)
    
    for index2, item2 in settings.dynamic_obj_dict.items():
        obj = bpy.data.objects[index2]
        for frame, data in item2.items():
            controller.log(f"Setze FINALEN Keyframe für {index2} bei Frame {frame}: Position {data.get('variante1')} Rotation {data.get('variante2')}")





def get_all_agents():
    agents = []
    for name, sub in settings.obj_info_dict.items():
        state0 = sub['state_dict'].get(0, {})
        if state0.get('hindernisArt') == 'agent':
            obj = sub['obj']
            agents.append(obj)
    return agents

def get_all_obstacles():
    obstacles = []
    for name, sub in settings.obj_info_dict.items():
        state0 = sub['state_dict'].get(0, {})
        if (state0.get('hindernisArt') == 'statisch' or state0.get('hindernisArt') == 'dynamisch') and state0.get('hindernisAktiv') == 'ja':
            obj = sub['obj']
            obstacles.append(obj)
    return obstacles

#test funktion, die bei value == 0 einen würfel in location2 erstellt
def addCube(location2, value1Or0, index):
    if value1Or0 == 0:
        bpy.ops.mesh.primitive_cube_add(location=location2, scale=(0.5, 0.5, 1.0))
        bpy.context.active_object.name = 'point' + str(index)
#listet alle hindernisse auf
def get_hindernis_dict(dynamischeHindernisseList):

    hindernisDict = {}
    i = 0
    for i, (index, item) in enumerate(settings.obj_info_dict.items()):
        if settings.obj_info_dict[index]['state_dict'][0]['hindernisAktiv'] == 'ja':
            if settings.obj_info_dict[index]['obj'].type == 'MESH':
                """mat = obj.matrix_world
                for vertex in settings.obj_info_dict[obj.name]['obj'].data.vertices:
                    loc = mat @ vertex.co
                    hindernisDict[i] = loc"""
                hindernisDict[i] = settings.obj_info_dict[index]['obj']
                if settings.obj_info_dict[index]['state_dict'][0]['hindernisArt'] == 'dynamisch':
                    dynamischeHindernisseList.append(settings.obj_info_dict[index]['obj'])
                i += 1
                
    return hindernisDict
# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    check_prop()
    get_state_txt_files()
    controller.init_controller()
    print('es hat funktioniert')
