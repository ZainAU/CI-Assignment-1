import random
from PIL import Image, ImageDraw
import colour
import matplotlib.pyplot
import numpy as np

import Q2.imgChromo as imgChromo
from EA_Base import EA
rng = np.random.default_rng()


class EvolutionaryLisa(EA):
    """

    """

    def __init__(self, img_size=(200, 200),
                 seed=rng,
                 population_size=30,
                 dataset=r"mona.png",
                 mutation_rate=0.5,
                 offspring_number=10,
                 num_generations=10,
                 iterations=10,
                 selection_method='FPS',
                 optimization_type='minimization',
                 mutation_type='insert'
                 ) -> None:

        super().__init__(seed, population_size, dataset, mutation_rate, offspring_number,
                         num_generations, iterations, selection_method, mutation_type, optimization_type)
        self.dataset = self.filename
        original_image = Image.open(self.filename)
        self.target_image = original_image.resize((264, 305))
        # why did the original code use these specific values?
        # self.target_image = original_image.resize((200,200))
        self.img_size = (self.target_image.size[0], self.target_image.size[1])
        imgChromo.ImgChromo(self.target_image.size)  # initialize image

    def get_fitness(self, chromosome, target):
        lab1 = np.array(chromosome.array)
        lab2 = np.array(target)
        self.fitness = np.mean(
            colour.difference.delta_E_CIE1976(lab1, lab2))  # delta_E_CIE1976(target, self.array))

    def population_init(self):
        filename = self.dataset
        for i in range(self.population_size):
            newChromosome = imgChromo(self.img_size)
            # sets newChromosome.fitness
            self.get_fitness(newChromosome, self.target_image)
            self.population.append(newChromosome)

    def evolutionary_process(self, population_size, generations):
        '''

        '''
        params = {'generations': [],
                  'fitness_estimate': [],
                  'crossover_mode': [],
                  'population_generation_used': [],
                  'im_size': []}

        # One of three solutions below
        # population = []
        self.population_init()
        # def population_init() where EvolutionaryLisa inherits from EA
        for i in range(self.num_generations):
            new_pop = []  # this is not init pop? What is this?
            # fittest_estimate = float('inf') // we calculate this later right?
            # why not replace with population_size and a simple int counter, since len operations take longer
            while len(new_pop) < len(self.population):
                
