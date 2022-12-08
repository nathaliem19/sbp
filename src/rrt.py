import numpy as np

from .map import Map
from .tree import Tree


def segment_is_free(mp: Map, coord1: np.ndarray, coord2: np.ndarray,
                    resolution=1) -> bool:
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    for step in range(abs(x1 - x2) // resolution):
        x = min(x1, x2) + step * resolution
        y = min(y1, y2) + step * resolution * abs((y2 - y1)/(x2-x1))
        x = int(x)
        y = int(y)
        if mp.data[x, y] == 1:
            return False
    return True


def rrt(mp: Map, iterations: int):
    eps = 20
    tree = Tree(mp.start)
    for _ in range(iterations):
        point = mp.sample()
        nearest, nearest_id = tree.get_nearest(point)
        new_point = nearest + eps * (point - nearest)/np.linalg.norm(point - nearest)
        new_point[0], new_point[1] = int(new_point[0]), int(new_point[1])
        new_point = new_point.astype(int)
        if segment_is_free(mp, new_point, nearest):
            tree.add_leaf(new_point, nearest_id)

            if mp.data[new_point[0], new_point[1]] == 2:
                v_id = len(tree.vertices)-1
                tree.path = []
                while v_id >= 0:
                    tree.path.append(v_id)
                    v_id = tree.parent[v_id]
                return tree

    return tree

