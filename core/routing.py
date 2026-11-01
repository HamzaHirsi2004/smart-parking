import math

# 1 map unit ≈ one bay width. Used only to show an approximate distance.
METRES_PER_UNIT = 2.5


def find_nearest(spaces, entrance_xy, n=3):
    """Return the n closest free bays to the entrance, nearest first."""
    free = []
    for s in spaces:
        if not s["is_occupied"]:
            d = math.dist((s["map_x"], s["map_y"]), entrance_xy)
            free.append({**s, "distance_m": round(d * METRES_PER_UNIT)})
    return sorted(free, key=lambda s: s["distance_m"])[:n]