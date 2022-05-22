import PIL
import random
import numpy as np
from straug.warp import Curve, Stretch, Distort
from straug.geometry import Perspective, Rotate, Shrink
from straug.pattern import Grid, VGrid, HGrid, RectGrid, EllipseGrid
from straug.blur import GaussianBlur, DefocusBlur,MotionBlur, GlassBlur, ZoomBlur 
from straug.process import Invert
from straug.camera import Contrast, Pixelate
from straug.weather import Fog, Snow, Frost, Rain, Shadow
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise


def augment_list():
    rng = np.random.default_rng(23)
    aug_list = [ #Stretch(rng), #Distort(rng), #Curve(rng=rng),
#                Perspective(rng), Shrink(rng), # Rotate(rng=rng),
                Grid(rng), VGrid(rng), HGrid(rng), RectGrid(rng),
                EllipseGrid(rng), Invert(rng), Contrast(rng), Pixelate(rng),
                Fog(rng), Snow(rng), Frost(rng), Rain(rng), Shadow(rng),
                GaussianNoise(rng), ShotNoise(rng), ImpulseNoise(rng), SpeckleNoise(rng)]

    return aug_list


class RandAugment(object):
    def __init__(self, n=1, p=0.3):
        self.num_select = n
        self.p = p
        self.augment_list = augment_list()

    def __call__(self, image):
        if np.random.rand() < self.p:
            mag = np.random.randint(-1, 3)
            ops = random.choices(self.augment_list, k=self.num_select)
            for op in ops:
                if isinstance(op, (Fog, Snow, Frost, Rain, Shadow)) and (image.size[0] <= 10 or image.size[1] <= 10):
                    continue
                image = op(image, mag=mag)

        return image

