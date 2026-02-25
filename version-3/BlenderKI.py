# BlenderKI.py
# Zentrale KI-Funktionen: Objekt-Initialisierung, State-Handling, A*-Navigation

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


import math
if IN_BLENDER:
    from mathutils import Vector, Euler, Quaternion
    import mathutils
else:
    # Dummy‑Klassen für Schritt 2 (werden nie benutzt)
    Vector = None
    Euler = None
    Quaternion = None
    mathutils = None
import os.path
import time, sys, os
import heapq
import json 

if IN_BLENDER:
    BASE_DIR = os.path.dirname(bpy.data.filepath)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
    
OBJINFO_JSON_PATH = os.path.join(BASE_DIR, "obj_info_dict.json")



import settings
import controller

float = __builtins__["float"]

class Vec:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_tuple(cls, t):
        return cls(t[0], t[1], t[2])

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def normalized(self):
        l = self.length()
        if l == 0:
            return Vec(0,0,0)
        return Vec(self.x/l, self.y/l, self.z/l)

def travel_time_in_frames(oldLocation, newLocation, speed_cm_per_sec):
    # oldLocation kann Tuple oder Vec sein
    if isinstance(oldLocation, Vec):
        old_loc = oldLocation
    else:
        old_loc = Vec(oldLocation[0], oldLocation[1], oldLocation[2])

    # newLocation kann Tuple oder Vec sein
    if isinstance(newLocation, Vec):
        new_loc = newLocation
    else:
        new_loc = Vec(newLocation[0], newLocation[1], newLocation[2])

    dx = new_loc.x - old_loc.x
    dy = new_loc.y - old_loc.y
    dz = new_loc.z - old_loc.z
    dist1 = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    distance_cm = dist1 * 100.0  # Blender arbeitet in Metern → Umrechnung in cm
    fps = controller.external_scene_data["fps"]

    time_seconds = distance_cm / speed_cm_per_sec
    frames = time_seconds * fps

    return frames

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
    update_progress("Aufgabe 1: check_prop", 0)
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
                'obj': None, #ob, #WICHTIG
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
    """
    Ersetzt die alte TXT-Logik:
    - Wenn obj_info_dict.json existiert:
        -> Laden, in settings.obj_info_dict übernehmen,
           'obj' auf echte Blender-Objekte setzen.
    - Wenn nicht:
        -> Aktuelles settings.obj_info_dict (aus check_prop) als JSON speichern.
    In JSON und obj_info_dict werden nur JSON-kompatible Typen gespeichert
    (keine mathutils-Objekte, keine bpy-Objekte).
    """
    obj_count = len(settings.obj_info_dict.items())
    update_progress("Aufgabe 2: get_state_json", 0)

    # Fall 1: JSON existiert → laden
    if os.path.exists(OBJINFO_JSON_PATH):
        with open(OBJINFO_JSON_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # In settings.obj_info_dict übernehmen
        settings.obj_info_dict.clear()
        izahl = 0
        for obj_name, sub in loaded.items():
            izahl += 1
            update_progress("Aufgabe 2: get_state_json", ((1.0 / max(obj_count, 1)) * izahl))
            # Sicherstellen, dass alle Schlüssel existieren
            state_dict = sub.get("state_dict", {})
            keyframes_dict = sub.get("keyframes_dict", {})
            variablen = sub.get("variablen", {})
            konstrukt_code = sub.get("konstrukt_code", {})

            # 'obj' wird NICHT aus JSON geladen, sondern hier gesetzt
            obj_ref = bpy.data.objects.get(obj_name, None)

            settings.obj_info_dict[obj_name] = {
                "obj": obj_ref,
                "state_dict": state_dict,
                "keyframes_dict": keyframes_dict,
                "variablen": variablen,
                "konstrukt_code": konstrukt_code
            }

        update_progress("Aufgabe 2: get_state_json", 1)
        return

    # Fall 2: JSON existiert nicht → aktuelles obj_info_dict als Basis nehmen
    # check_prop() muss vorher gelaufen sein
    # 'obj' wird vor dem Speichern auf None gesetzt, damit JSON sauber bleibt
    serializable_dict = {}
    for i, (obj_name, sub) in enumerate(settings.obj_info_dict.items()):
        update_progress("Aufgabe 2: get_state_json", ((1.0 / max(obj_count, 1)) * i))

        state_dict = sub.get("state_dict", {})
        keyframes_dict = sub.get("keyframes_dict", {})
        variablen = sub.get("variablen", {})
        konstrukt_code = sub.get("konstrukt_code", {})

        serializable_dict[obj_name] = {
            "obj": None,  # Blender-Objekte nicht in JSON speichern
            "state_dict": state_dict,
            "keyframes_dict": keyframes_dict,
            "variablen": variablen,
            "konstrukt_code": konstrukt_code
        }

    with open(OBJINFO_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_dict, f, indent=4)

    # Nach dem Speichern: 'obj' in settings.obj_info_dict auf echte Objekte setzen
    for obj_name, sub in settings.obj_info_dict.items():
        sub["obj"] = bpy.data.objects.get(obj_name, None)

    update_progress("Aufgabe 2: get_state_json", 1)

# ----------------------------------------------------------
# Hilfszugriffe auf Szenedaten (nur Namen, keine Objekte)
# ----------------------------------------------------------

def _get_object_by_name(name):
    """Liefert das Objekt-Dict aus external_scene_data['objects'] anhand des Namens."""
    for o in controller.external_scene_data["objects"]:
        if o["name"] == name:
            return o
    return None

def _get_location(name):
    """Gibt die Weltposition (x,y,z) eines Objekts als Tupel zurück."""
    obj = _get_object_by_name(name)
    if obj is None:
        return (0.0, 0.0, 0.0)
    loc = obj["location"]
    return (float(loc[0]), float(loc[1]), float(loc[2]))

def _get_bbox_points(name):
    """Gibt die 8 Bounding-Box-Punkte eines Objekts als Liste von (x,y,z)-Tupeln zurück."""
    bbox = controller.external_scene_data["bbox"].get(name, [])
    return [ (float(p[0]), float(p[1]), float(p[2])) for p in bbox ]

def _get_vertices_worldspace_list(name):
    """Gibt die Welt-Vertices eines Objekts als Liste von (x,y,z)-Tupeln zurück."""
    verts = controller.external_scene_data["vertices"].get(name, [])
    return [ (float(v[0]), float(v[1]), float(v[2])) for v in verts ]

# ----------------------------------------------------------
# Bounding Box / Grid / Hindernisse (nur Namen)
# ----------------------------------------------------------

CELL_SIZE = 0.75
GRID_MARGIN = 2.0
OBSTACLE_BUFFER = 0.3

def bbox_min_max(obj_name):
    """Min/Max der Bounding Box eines Objekts (nur über Namen)."""
    corners = _get_bbox_points(obj_name)
    if not corners:
        return ( (0.0, 0.0), (0.0, 0.0), (0.0, 0.0) )

    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return ( (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)) )

def point_in_bbox_2d_with_buffer(x, y, obj_name, buffer):
    (minx, maxx), (miny, maxy), _ = bbox_min_max(obj_name)
    return (minx - buffer <= x <= maxx + buffer) and (miny - buffer <= y <= maxy + buffer)

def world_to_grid(x, y, min_x, min_y):
    gx = int((x - min_x) / CELL_SIZE)
    gy = int((y - min_y) / CELL_SIZE)
    return gx, gy

def grid_to_world(gx, gy, min_x, min_y, z):
    x = min_x + gx * CELL_SIZE + CELL_SIZE * 0.5
    y = min_y + gy * CELL_SIZE + CELL_SIZE * 0.5
    return (x, y, z)

def is_cell_blocked(gx, gy, min_x, min_y, obstacles, buffer=0.0):
    x = min_x + gx * CELL_SIZE + CELL_SIZE * 0.5
    y = min_y + gy * CELL_SIZE + CELL_SIZE * 0.5
    for ob_name in obstacles:
        if point_in_bbox_2d_with_buffer(x, y, ob_name, buffer):
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

def build_grid_bounds(agents, goal_positions, obstacles):
    """
    agents: Liste von Agent-Namen
    goal_positions: Liste von (x,y,z)
    obstacles: Liste von Hindernis-Namen
    """
    xs, ys = [], []

    for ag_name in agents:
        x, y, _ = _get_location(ag_name)
        xs.append(x)
        ys.append(y)

    for gp in goal_positions:
        xs.append(gp[0])
        ys.append(gp[1])

    for ob_name in obstacles:
        (minx, maxx), (miny, maxy), _ = bbox_min_max(ob_name)
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])

    if not xs or not ys:
        return -GRID_MARGIN, GRID_MARGIN, -GRID_MARGIN, GRID_MARGIN

    min_x = min(xs) - GRID_MARGIN
    max_x = max(xs) + GRID_MARGIN
    min_y = min(ys) - GRID_MARGIN
    max_y = max(ys) + GRID_MARGIN

    return min_x, max_x, min_y, max_y

# ----------------------------------------------------------
# Agenten / Hindernisse (nur Namen)
# ----------------------------------------------------------

def get_agent_width(agent_name):
    (minx, maxx), (miny, maxy), _ = bbox_min_max(agent_name)
    return max(maxx - minx, maxy - miny)

def get_all_agents():
    """
    Liefert alle Agenten-Namen (hindernisArt == 'agent').
    """
    agents = []
    for name, sub in settings.obj_info_dict.items():
        state0 = sub['state_dict'].get(0) or sub['state_dict'].get("0", {})
        if state0.get('hindernisArt') == 'agent':
            agents.append(name)
    return agents

def get_all_obstacles():
    """
    Liefert alle Hindernis-Namen (statisch oder dynamisch, hindernisAktiv == 'ja').
    """
    obstacles = []
    for name, sub in settings.obj_info_dict.items():
        state0 = sub['state_dict'].get(0) or sub['state_dict'].get("0", {})
        if (state0.get('hindernisArt') in ('statisch', 'dynamisch')
                and state0.get('hindernisAktiv') == 'ja'):
            obstacles.append(name)
    return obstacles


# ----------------------------------------------------------
# AABB-basierter "Raycast" zwischen Agenten (nur Geometrie)
# ----------------------------------------------------------

def agent_collision_raycast(world_pos, vertices_list_obj_dict, self_name, distance=1.0):
    """
    world_pos: (x,y,z)
    vertices_list_obj_dict: {obj_name: [(x,y,z), ...], ...}
    self_name: Name des Agents
    distance: float
    """

    controller.log(f"[COL] {self_name}: world_pos={world_pos}, dist={distance}, objs={list(vertices_list_obj_dict.keys())}")
    
    # Sichere Richtungsvektoren (falls global etwas überschrieben wurde)
    # 8 Strahlen (2 pro Seite)
    safe_dirs = []
    base_dirs = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
    ]

    for d in base_dirs:
        safe_dirs.append(d)
        safe_dirs.append((d[0], d[1], 0.1))
        safe_dirs.append((d[0], d[1], -0.1))

    # AABBs vorberechnen
    obj_bounds = {}
    for obj_name, verts in vertices_list_obj_dict.items():
        if obj_name == self_name:
            continue
        if not verts:
            continue

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]

        obj_bounds[obj_name] = {
            "min": (min(xs), min(ys), min(zs)),
            "max": (max(xs), max(ys), max(zs)),
        }

    # Lokale AABB-Funktion (Name bewusst eindeutig)
    def _point_in_aabb_local(point, aabb_dict):
        x, y, z = point
        minx, miny, minz = aabb_dict["min"]
        maxx, maxy, maxz = aabb_dict["max"]
        return (
            minx <= x <= maxx and
            miny <= y <= maxy and
            minz <= z <= maxz
        )

    # Sichere Schrittanzahl
    safe_steps = 10
    try:
        safe_steps = int(safe_steps)
    except:
        safe_steps = 10

    # Sichere Schrittweite
    try:
        step_len = float(distance) / float(safe_steps)
    except:
        step_len = 0.1

    # world_pos kann Tuple oder Vec sein
    if isinstance(world_pos, Vec):
        px, py, pz = world_pos.x, world_pos.y, world_pos.z
    else:
        px, py, pz = world_pos

    # Hauptschleife
    for dx, dy, dz in safe_dirs:

        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length == 0:
            continue

        ndx, ndy, ndz = dx/length, dy/length, dz/length

        for step_index in range(1, safe_steps + 1):

            cx = px + ndx * (step_index * step_len)
            cy = py + ndy * (step_index * step_len)
            cz = pz + ndz * (step_index * step_len)

            check_point = (cx, cy, cz)

            for obj_name, aabb_dict in obj_bounds.items():
                if _point_in_aabb_local(check_point, aabb_dict):
                    controller.log(f"[COL] {self_name}: Kollision bei {check_point} mit {obj_name}")
                    return True

    return False



# ----------------------------------------------------------
# Pfad → Weltkoordinaten (ohne mathutils)
# ----------------------------------------------------------

def grid_path_to_world(path_g, min_x, min_y, z):
    """
    path_g: [((gx,gy), t), ...]
    Rückgabe: [((x,y,z), t), ...]
    """
    pts = []
    for (gx, gy), t in path_g:
        x, y, z0 = grid_to_world(gx, gy, min_x, min_y, z)
        pts.append(((x, y, z0), t))
    return pts

def get_vertices_worldspace(obj):
    """
    Gibt eine Liste aller Vertex-Positionen eines Mesh-Objekts in Weltkoordinaten zurück.
    Wird nur in Schritt 1 (Blender) verwendet.
    """
    if not IN_BLENDER:
        raise RuntimeError("get_vertices_worldspace darf nur in Blender (Schritt 1) verwendet werden.")

    if obj.type != 'MESH':
        return []

    world_matrix = obj.matrix_world
    mesh = obj.data
    return [ (world_matrix @ v.co).to_tuple() for v in mesh.vertices ]

def world_bbox_corners(obj):
    """
    Gibt die 8 Bounding-Box-Eckpunkte eines Objekts in Weltkoordinaten zurück.
    Wird nur in Schritt 1 (Blender) verwendet.
    """
    if not IN_BLENDER:
        raise RuntimeError("world_bbox_corners darf nur in Blender (Schritt 1) verwendet werden.")

    mat = obj.matrix_world
    return [ (mat @ mathutils.Vector(corner)).to_tuple() for corner in obj.bound_box ]


# ----------------------------------------------------------
# A* mit Zeitdimension (nur Namen, keine Objekte)
# ----------------------------------------------------------

def a_star(self_agent_name, all_agents_names, agent_width, start, goal,
           min_x, min_y, max_x, max_y, obstacles_names, reserved=None, buffer=0.0):
    """
    self_agent_name: Name des Agents
    all_agents_names: Liste von Agent-Namen
    start/goal: (gx,gy)
    reserved: dict[(gx,gy,t)] = True
    """
    controller.log(f"[A*] START {self_agent_name}, start={start}, goal={goal}")
    max_iters = 50000
    iters = 0

    if reserved is None:
        reserved = {}

    start_g = start
    goal_g = goal

    open_set = []
    heapq.heappush(open_set, (0, start_g, 0))  # (f, (gx,gy), time_step)
    came_from = {}
    g_score = {(start_g, 0): 0}

    max_gx = int((max_x - min_x) / CELL_SIZE) + 1
    max_gy = int((max_y - min_y) / CELL_SIZE) + 1

    while open_set:
        iters += 1
        if iters % 1000 == 0:
            controller.log(f"[A*] {self_agent_name}: Iteration {iters}, open={len(open_set)}")

        if iters > max_iters:
            controller.log(f"[A*] {self_agent_name}: Abbruch, max_iters erreicht")
            return None


        _, current, t = heapq.heappop(open_set)

        if current == goal_g:
            path = []
            key = (current, t)
            while key in came_from:
                (cell, time_step) = key
                path.append((cell, time_step))
                key = came_from[key]
            (cell, time_step) = key
            path.append((cell, time_step))
            path.reverse()
            controller.log(f"[A*] {self_agent_name}: Pfad-Länge = {len(path)}")
            return path
        for nx, ny in neighbors(*current):
            if nx < 0 or ny < 0 or nx >= max_gx or ny >= max_gy:
                continue

            wx, wy, _ = grid_to_world(nx, ny, min_x, min_y, 0.0)
            sx, sy, sz = _get_location(self_agent_name)
            world_pos = (wx, wy, sz)

            # Vertices aller Agenten vorbereiten
            vertices_list_obj_dict = {}
            for a_name in all_agents_names:
                vertices_list_obj_dict[a_name] = _get_vertices_worldspace_list(a_name)

            if agent_collision_raycast(world_pos, vertices_list_obj_dict, self_agent_name, distance=agent_width):
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

            if is_cell_blocked(nx, ny, min_x, min_y, obstacles_names, buffer=buffer):
                continue

            nt = t + 1

            if reserved.get((nx, ny, nt), False):
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
    
    controller.log(f"[A*] {self_agent_name}: Kein Pfad gefunden (Grund X)")
    return None


# ----------------------------------------------------------
# Pfadverteilung / Sortierung (nur Namen)
# ----------------------------------------------------------

def sort_agents_by_last_waypoint_time(all_paths_world, agents):
    """
    all_paths_world: {agent_name: [((x,y,z), t), ...]}
    agents: Liste von Agent-Namen
    """
    def last_t(agent_name):
        path = all_paths_world.get(agent_name, [])
        if not path:
            return float('inf')
        return path[-1][1]

    return sorted(agents, key=last_t)

def distribute_goal_positions(global_target_name, agents, spacing=1.0):
    """
    global_target_name: Name des Zielobjekts
    agents: Liste von Agent-Namen
    Rückgabe: Liste von (x,y,z)
    """
    tx, ty, tz = _get_location(global_target_name)
    count = len(agents)

    if count == 0:
        return []

    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    positions = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= count:
                break
            offset_x = (c - cols/2) * spacing
            offset_y = -(r * spacing)
            positions.append((tx + offset_x, ty + offset_y, tz))
            idx += 1

    return positions


# ----------------------------------------------------------
# Bewegungs-/Rotations-Dictionaries (nur Listen/Tupel)
# ----------------------------------------------------------

def expand_move_dict(moveDict):
    """
    moveDict: {frame: (x,y,z)}
    Rückgabe: {frame: (x,y,z)} mit interpolierten Zwischenframes
    """
    frames = sorted(moveDict.keys())
    if not frames:
        return {}

    newDict = {}

    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end = frames[i+1]

        v_start = moveDict[f_start]
        v_end = moveDict[f_end]

        newDict[f_start] = v_start

        frame_range = f_end - f_start
        if frame_range <= 0:
            continue

        x1, y1, z1 = v_start
        x2, y2, z2 = v_end

        for f in range(f_start + 1, f_end):
            t = (f - f_start) / frame_range
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            z = z1 + (z2 - z1) * t
            newDict[f] = (x, y, z)

    last_frame = frames[-1]
    newDict[last_frame] = moveDict[last_frame]

    return newDict

def expand_rotation_dict(rotDict):
    """
    rotDict: {frame: (rx,ry,rz)} in Radiant
    Rückgabe: {frame: (rx,ry,rz)} mit linear interpolierten Zwischenframes
    """
    frames = sorted(rotDict.keys())
    if not frames:
        return {}

    newDict = {}

    for i in range(len(frames) - 1):
        f_start = frames[i]
        f_end = frames[i+1]

        r_start = rotDict[f_start]
        r_end = rotDict[f_end]

        newDict[f_start] = r_start

        frame_range = f_end - f_start
        if frame_range <= 0:
            continue

        rx1, ry1, rz1 = r_start
        rx2, ry2, rz2 = r_end

        for f in range(f_start + 1, f_end):
            t = (f - f_start) / frame_range
            rx = rx1 + (rx2 - rx1) * t
            ry = ry1 + (ry2 - ry1) * t
            rz = rz1 + (rz2 - rz1) * t
            newDict[f] = (rx, ry, rz)

    last_frame = frames[-1]
    newDict[last_frame] = rotDict[last_frame]

    return newDict


# ----------------------------------------------------------
# Freie Richtung (nur Geometrie, keine Objekte)
# ----------------------------------------------------------

def compute_free_direction_geometric(pos, agents, self_name, radius):
    """
    pos: (x,y,z)
    agents: Liste von Agent-Namen
    self_name: Name des eigenen Agents
    radius: Suchradius
    """
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
    ]
    if isinstance(pos, Vec):
        px, py, pz = pos.x, pos.y, pos.z
    else:        
        px, py, pz = pos
    best_dir = (0.0, 0.0, 0.0)
    best_dist = -1.0

    for dx, dy, dz in directions:
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length == 0:
            continue
        ndx, ndy, ndz = dx/length, dy/length, dz/length

        test_pos = (px + ndx * radius, py + ndy * radius, pz + ndz * radius)
        tx, ty, tz = test_pos

        min_dist = 9999.0

        for ag_name in agents:
            if ag_name == self_name:
                continue
            ax, ay, az = _get_location(ag_name)
            dist = math.sqrt((tx - ax)**2 + (ty - ay)**2 + (tz - az)**2)
            if dist < min_dist:
                min_dist = dist

        if min_dist > best_dist:
            best_dist = min_dist
            best_dir = (ndx, ndy, ndz)

    return best_dir


# ----------------------------------------------------------
# Hauptfunktion: compute_astar_world_path (nur Namen)
# ----------------------------------------------------------

def compute_astar_world_path(ziel_obj_name, agents, obstaclesList,
                             agentenDict, start_frame, end_frame,
                             agentenDictMitHindernissen, current_state):
    """
    ziel_obj_name: Name des Zielobjekts
    agents: Liste von Agent-Namen
    obstaclesList: Liste von Hindernis-Namen
    agentenDict: {index: {..., 'name': agent_name, 'speed':..., 'zeit_aktiv':..., 'time':...}}
    """
    controller.log("test1")
    data = controller.external_scene_data
    fps = data["fps"]
    base_frame = data["current_frame"]

    global_target_name = ziel_obj_name
    obstacles = obstaclesList

    if not agents:
        controller.log("Keine Agenten gefunden.")
        return
    controller.log("test2")

    if not global_target_name:
        controller.log("Globales Zielobjekt nicht gefunden.")
        return

    obj_width_list = [get_agent_width(a) for a in agents]
    if not obj_width_list:
        controller.log("Keine Agentenbreiten gefunden.")
        return

    max_width = max(obj_width_list)
    goal_spacing = max_width + 0.5

    # PHASE 1
    controller.log("test3")
    goal_positions_initial = distribute_goal_positions(global_target_name, agents, spacing=goal_spacing)
    min_x, max_x, min_y, max_y = build_grid_bounds(agents, goal_positions_initial, obstacles)
    controller.log("test3a")
    reserved = {}
    all_paths_world = {}

    update_progress("Aufgabe 4: compute_astar_world_path: phase 1", 0)
    for agent_name, goal_pos in zip(agents, goal_positions_initial):
        update_progress("Aufgabe 4: compute_astar_world_path: phase 1", ((1.0 / len(agents)) * (agents.index(agent_name) + 1)))
        sx, sy, sz = _get_location(agent_name)
        start_g = world_to_grid(sx, sy, min_x, min_y)
        gx, gy, gz = goal_pos
        goal_g = world_to_grid(gx, gy, min_x, min_y)
        controller.log("test3b")
        path_g = a_star(
            agent_name, agents, max_width,
            start_g, goal_g,
            min_x, min_y, max_x, max_y,
            obstacles,
            reserved=reserved,
            buffer=OBSTACLE_BUFFER
        )
        
        controller.log("test3c: Agent {0}, Pfad gefunden: {1}".format(agent_name, path_g is not None))
        if path_g is None:
            continue

        for (cx, cy), t in path_g:
            reserved[(cx, cy, t)] = True

        world_path = grid_path_to_world(path_g, min_x, min_y, sz)
        all_paths_world[agent_name] = world_path
    update_progress("Aufgabe 4: compute_astar_world_path: phase 1", 1)
        
    controller.log("test3d: Alle Pfade berechnet, Anzahl: {0}".format(len(all_paths_world)))

    # PHASE 2
    agents_sorted = sort_agents_by_last_waypoint_time(all_paths_world, agents)
    
    controller.log("test3e: Agenten sortiert: {0}".format(agents_sorted))

    # PHASE 3
    goal_positions_final = distribute_goal_positions(global_target_name, agents_sorted, spacing=goal_spacing)

    # PHASE 4
    reserved = {}
    all_paths_world_final = {}
    
    controller.log("test4")
    update_progress("Aufgabe 5 : compute_astar_world_path: phase 4", 0)
    for agent_name, goal_pos in zip(agents_sorted, goal_positions_final):
        update_progress("Aufgabe 5 : compute_astar_world_path: phase 4", ((1.0 / len(agents_sorted)) * (agents_sorted.index(agent_name) + 1)))
        sx, sy, sz = _get_location(agent_name)
        start_g = world_to_grid(sx, sy, min_x, min_y)
        gx, gy, gz = goal_pos
        goal_g = world_to_grid(gx, gy, min_x, min_y)

        path_g = a_star(
            agent_name, agents_sorted, max_width,
            start_g, goal_g,
            min_x, min_y, max_x, max_y,
            obstacles,
            reserved=reserved,
            buffer=OBSTACLE_BUFFER
        )

        if path_g is None:
            continue

        for (cx, cy), t in path_g:
            reserved[(cx, cy, t)] = True

        world_path = grid_path_to_world(path_g, min_x, min_y, sz)
        all_paths_world_final[agent_name] = world_path

        # Navigationspfad in obj_info_dict speichern (nur Zahlen)
        if agent_name in settings.obj_info_dict:
            settings.obj_info_dict[agent_name]["navigationsPfad"] = [
                (float(p[0][0]), float(p[0][1]), float(p[0][2]), int(p[1]))
                for p in world_path
            ]
    update_progress("Aufgabe 5 : compute_astar_world_path: phase 4", 1)
    controller.log("test5")
    # PHASE 5: moveDict / rotDict über controller.animate_agent berechnen
    update_progress("Aufgabe 6 : compute_astar_world_path: phase 5", 0)
    for index2, item2 in agentenDict.items():
        update_progress("Aufgabe 6 : compute_astar_world_path: phase 5", ((1.0 / len(agentenDict)) * (index2 + 1)))
        agent_name = item2['name']
        world_path = all_paths_world_final.get(agent_name)
        if not world_path:
            continue

        moveDict, rotDict = controller.animate_agent(
            agent_name,
            world_path,
            fps,
            base_frame,
            item2['speed'],
            item2['zeit_aktiv'],
            item2['time']
        )

        agentenDict[index2]['moveDictNeu'] = moveDict
        agentenDict[index2]['rotDictNeu'] = rotDict
    update_progress("Aufgabe 6 : compute_astar_world_path: phase 5", 1)  
    for index2, item2 in agentenDict.items():
        agentenDict[index2]['moveDictNeu'] = expand_move_dict(item2['moveDictNeu'])
        agentenDict[index2]['rotDictNeu'] = expand_rotation_dict(item2['rotDictNeu'])
        
    # ----------------------------------------------------------
    # PHASE 6: Kollisionsvermeidung durch "Wegschubsen" implementieren
    # ----------------------------------------------------------
    
    # 3 Wiederholungen
    update_progress("Aufgabe 7 : compute_astar_world_path: phase 6 a (3 wiederholungen)", 0)
    for _ in range(3):
        update_progress("Aufgabe 7 : compute_astar_world_path: phase 6 a (3 wiederholungen)", ((1.0 / 3) * (_ + 1)))
        # Alle Frames durchgehen
        update_progress("Aufgabe 8 : compute_astar_world_path: phase 6 b (frames)", 0)
        for frame in range(start_frame, end_frame + 1):
            update_progress("Aufgabe 8 : compute_astar_world_path: phase 6 b (frames)", ((1.0 / (end_frame - start_frame + 1)) * (frame)))
            # Alle Agenten prüfen
            update_progress("Aufgabe 9 : compute_astar_world_path: phase 6 c (agenten)", 0)
            for index2, item2 in agentenDict.items():
                update_progress("Aufgabe 9 : compute_astar_world_path: phase 6 c (agenten)", ((1.0 / len(agentenDict)) * (index2 + 1)))
                if frame not in item2['moveDictNeu']:
                    continue

                pos = Vec.from_tuple(item2['moveDictNeu'][frame])
                self_obj_name = item2['name']
                self_name = self_obj_name
                agent_width = get_agent_width(self_obj_name)
                min_dist = agent_width * 0.6
                ray_dist = 0.5

                # 8 Strahlen (2 pro Seite)
                ray_dirs = []
                base_dirs = [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (1, 1, 0),
                    (1, -1, 0),
                    (-1, 1, 0),
                    (-1, -1, 0),
                ]

                for d in base_dirs:
                    ray_dirs.append(d)
                    ray_dirs.append((d[0], d[1], 0.1))
                    ray_dirs.append((d[0], d[1], -0.1))

                collision_detected = False

                # Alle anderen Objekte prüfen
                update_progress("Aufgabe 10 : compute_astar_world_path: phase 6 d (kollision check)", 0)
                for other_index, other_val in agentenDictMitHindernissen.items():
                    update_progress("Aufgabe 10 : compute_astar_world_path: phase 6 d (kollision check)", ((1.0 / len(agentenDictMitHindernissen)) * (other_index + 1)))
                    # Hindernisse haben keine moveDictNeu → überspringen
                    if 'moveDictNeu' not in other_val:
                        continue

                    other_obj_name = other_val['name']
                    if other_obj_name == self_name:
                        continue
                    
                    if frame not in other_val['moveDictNeu'] and other_val['von_obj'] is not None:
                        continue
                    
                    # Position des anderen Objekts für Distanzcheck
                    if frame in other_val['moveDictNeu']:
                        other_pos = Vec.from_tuple(other_val['moveDictNeu'][frame])
                    else:
                        if settings.obj_info_dict[other_obj_name]["state_dict"][current_state]["hindernisArt"] == "statisch":
                            target = next((o for o in controller.external_scene_data["objects"] if o["name"] == other_obj_name), None)
                            other_pos = Vec(target["location"][0], target["location"][1], target["location"][2])
                    # Distanzcheck
                    dx = pos.x - other_pos.x
                    dy = pos.y - other_pos.y
                    dz = pos.z - other_pos.z
                    dist1 = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if dist1 < min_dist:
                        collision_detected = True
                        
                    # Vertices aller Agenten vorbereiten
                    vertices_list_obj_dict = {}
                    agent_and_obstacle_names = agents + obstacles
                    for a_name in agents:
                        vertices_list_obj_dict[a_name] = _get_vertices_worldspace_list(a_name)
                    for o_name in obstacles:
                        vertices_list_obj_dict[o_name] = _get_vertices_worldspace_list(o_name)
                

                    # Geometrie‑Raycast
                        if agent_collision_raycast(pos, vertices_list_obj_dict, self_obj_name, distance=ray_dist):
                            collision_detected = True
                            break

                    if collision_detected:
                        break
                
                update_progress("Aufgabe 10 : compute_astar_world_path: phase 6 d (kollision check)", 1)
                # Wenn keine Kollision → weiter
                if not collision_detected:
                    continue

                
                # Richtung mit freiem Raum bestimmen
                push_dir = Vec.from_tuple(
                    compute_free_direction_geometric(
                        pos,
                        agent_and_obstacle_names,
                        self_obj_name,
                        radius=min_dist
                    )
                ).normalized()
                penetration = (min_dist - ray_dist)
                push_amount = min(penetration, 0.1)  # max 10 cm pro Frame
                new_pos = pos + push_dir * push_amount
                while agent_collision_raycast(new_pos, vertices_list_obj_dict, self_obj_name, distance=ray_dist):
                    push_dir = Vec.from_tuple(
                        compute_free_direction_geometric(
                            pos,
                            agent_and_obstacle_names,
                            self_obj_name,
                            radius=min_dist * 0.5
                        )
                    ).normalized()
                    min_dist *= 0.5
                    if min_dist < 0.1:
                        break

                # Wegschubsen
                agentenDict[index2]['moveDictNeu'][frame] = new_pos.to_tuple()

                # Alle folgenden Frames verschieben
                sorted_frames = sorted(item2['moveDictNeu'].keys())
                push_key = travel_time_in_frames(pos, new_pos, item2['speed'])

                for f2 in sorted_frames:
                    if f2 > frame + push_key:
                        agentenDict[index2]['moveDictNeu'][f2] = item2['moveDictNeu'][f2] + push_dir * min_dist

                # Rotation anpassen
                if frame in item2['rotDictNeu']:
                    angle_z = controller.look_at_angle(pos, new_pos)
                    agentenDict[index2]['rotDictNeu'][frame] = (0.0, 0.0, angle_z)
                # Nach jedem Durchlauf expandieren
            for index2, item2 in agentenDict.items():
                agentenDict[index2]['moveDictNeu'] = expand_move_dict(item2['moveDictNeu'])
                agentenDict[index2]['rotDictNeu'] = expand_rotation_dict(item2['rotDictNeu'])
            update_progress("Aufgabe 9 : compute_astar_world_path: phase 6 c (agenten)", 1)
        update_progress("Aufgabe 8 : compute_astar_world_path: phase 6 b (frames)", 1)       
    update_progress("Aufgabe 7 : compute_astar_world_path: phase 6 a (3 wiederholungen)", 1)
    
    # ----------------------------------------------------------
    # PHASE 7: Keyframes in Blender setzen
    # ----------------------------------------------------------
    controller.log("test6")
    agentenDictName = ''
    for index2, item2 in agentenDict.items():
        controller.log("test7")
        # Name bestimmen
        if controller.external_scene_data is not None and not IN_BLENDER:
            name = item2['name']
            controller.log(f"test8 {name}")
        else:
            name = item2['obj'].name

        agentenDictName += name + ', '

        # === SCHRITT 2: Blender-freie Keyframe-Erzeugung ===
        if controller.external_scene_data is not None and not IN_BLENDER:

            # Positionen
            for frame, pos in item2['moveDictNeu'].items():
                controller.write_keyframes(item2['name'], frame, pos, None)

            # Rotationen
            for frame, rot in item2['rotDictNeu'].items():
                controller.write_keyframes(item2['name'], frame, None, rot)

        # === SCHRITT 3: Blender-Keyframes ===
        else:

            for frame, pos in item2['moveDictNeu'].items():
                controller.write_keyframes(item2['obj'], frame, pos, None)

            for frame, rot in item2['rotDictNeu'].items():
                controller.write_keyframes(item2['obj'], frame, None, rot)
            
            
        
    """         
    print("Agenten mit Pfaden: " + agentenDictName)
    
    for index2, item2 in settings.dynamic_obj_dict.items():
        for frame, data in item2.items():
            controller.log(f"Setze FINALEN Keyframe für {index2} bei Frame {frame}: Position {data.get('variante1')} Rotation {data.get('variante2')}")
    """

