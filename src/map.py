import numpy as np
from typing import Tuple
import plotly.express as px
import cv2
import random


class Map:
    def __init__(self, data: np.ndarray, start: Tuple[int, int]):
        '''
        0 -- free
        1 -- obsticle
        2 -- goal region
        '''
        self.data = data
        self.w = data.shape[0]
        self.h = data.shape[1]
        self.start = start

        random.seed(42)

    def sample(self) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            if self.data[x, y] == 0:
                return np.array([x, y])

    def to_image(self) -> np.ndarray:
        background_color = (0.8, 0.8, 0.9)
        obsticle_color = (0.2, 0.2, 0.2)
        start_color = (0.2, 1.0, 0.2)
        goal_color = (1.0, 0.34, 0.79)

        image = np.zeros((self.h, self.w, 3)) + background_color
        image[self.data.T == 1] = obsticle_color
        image[self.data.T == 2] = goal_color

        radius = int(min(self.w, self.h) / 100)
        image = cv2.circle(image, self.start, radius, color=start_color,
                           thickness=-1)
        image = cv2.circle(image, self.start, 2*radius, color=start_color,
                            thickness=radius//2)
        return image


def default_map() -> Map:
    data = np.zeros((1024, 800))
    data[500:600, 200:700] = 1
    data[800:1000, 550:750] = 2
    start = (150, 100)
    return Map(data, start)


def show_map(mp: Map):
    image = mp.to_image()
    fig = px.imshow(image)
    fig.show()

