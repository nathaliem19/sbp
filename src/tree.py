import numpy as np
from typing import Tuple
import cv2
import plotly.express as px

from .map import Map


class Tree:
    def __init__(self, root: np.ndarray):
        self.vertices = [root]
        self.parent = {0: -1}
        self.path = None

    def add_leaf(self, coord: np.ndarray, parent_id: int):
        self.vertices.append(coord)
        self.parent[len(self.vertices)-1] = parent_id

    def get_nearest(self, coord: np.ndarray) \
            -> Tuple[np.ndarray, int]:
        vertices = np.array(self.vertices)
        nearest_id = np.argmin(np.sum((vertices - coord)**2, axis=1))
        return self.vertices[nearest_id], nearest_id

    def _draw_edge(self, image: np.ndarray, v1_id: int, v2_id: int, width: int) \
            -> np.ndarray:
        coord1 = self.vertices[v1_id]
        coord2 = self.vertices[v2_id]
        return cv2.line(image, self.vertices[v1_id], self.vertices[v2_id],
                        color, thickness=width)


    def show_tree(self, image: np.ndarray) -> np.ndarray:
        color = (0.0, 0.0, 1.0)
        width = int(min(image.shape[0], image.shape[1])/200)
        for v_id, parent_id in self.parent.items():
            if parent_id >= 0:
                image = cv2.line(image,
                                 self.vertices[v_id],
                                 self.vertices[parent_id],
                                 color, thickness=width)

        if self.path is not None:
            path_color = (1.0, 0.0, 0.0)
            for i in range(len(self.path)-1):
                image = cv2.line(image,
                                 self.vertices[self.path[i]],
                                 self.vertices[self.path[i+1]],
                                 path_color, thickness=width)

        return image


def show_result(mp: Map, tree: Tree):
    image = mp.to_image()
    image = tree.show_tree(image)
    fig = px.imshow(image)
    fig.show()
