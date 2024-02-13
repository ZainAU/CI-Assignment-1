import random
from PIL import Image, ImageDraw
import colour
import matplotlib.pyplot
import numpy as np

import Q2.imgChromo as imgChromo
import EA_Base


class EvolutionaryLisa:
    """

    """

    def __init__(self, filename=r"mona.png") -> None:
        original_image = Image.open(filename)
        self.target_image = original_image.resize((264, 305))
        # why did the original code use these specific values?
        # self.target_image = original_image.resize((200,200))

        imgChromo.ImgChromo(self.target_image.size)

    def evolutionary_process(self, population_size, generations):
        '''

        '''
        params = {'generations': [],
                  'fitness_estimate': [],
                  'crossover_mode': [],
                  'population_generation_used': [],
                  'im_size': []}

        # initialize population should happen within EA class
        population = []
