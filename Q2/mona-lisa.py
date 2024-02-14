from pandas import DataFrame
import random
from PIL import Image, ImageDraw
import colour
import matplotlib.pyplot as plt
import numpy as np

from imgChromo import ImgChromo
from EA_Base import EA
import datetime
rng = np.random.default_rng()


class EvolutionaryLisa(EA):
    """

    """

    def __init__(self, img_size=(200, 200),
                 seed=rng,
                 population_size=30,
                 paramsset=r"mona.png",
                 mutation_rate=0.5,
                 offspring_number=10,
                 num_generations=10,
                 iterations=10,
                 parent_selection_method='FPS',
                 survival_selection_method='FPS',
                 optimization_type='minimization',
                 mutation_type='insert'
                 ) -> None:

        super().__init__(seed, population_size, paramsset, mutation_rate, offspring_number,
                         num_generations, iterations, parent_selection_method, survival_selection_method, mutation_type, optimization_type)
        self.filename = paramsset
        original_image = Image.open(self.filename)
        self.target_image = original_image.resize((264, 305))
        # why did the original code use these specific values?
        # self.target_image = original_image.resize((200,200))
        self.length, self.width = self.target_image.size
        self.img_size = (self.length, self.width)

        # initialize image
        ImgChromo(self.width, self.length)
        plt.imshow(self.target_image)

    # debugging selection processes

    def tournament_select(self, population, tournament_size=6):
        """
        Selects the most fit individual from a randomly sampled subset of the population

        Keyword arguments:
        population -- current generation's population
        tournament_size -- number of individuals randomly sampled to participate

        Returns:
        winner -- individual with the best fitness out of the tournament_size participants
        """

        # randomly sample participants
        indices = np.random.choice(len(population), tournament_size)
        random_subset = [population[i] for i in indices]

        winner = None

        # find individual with best fitness
        for i in random_subset:
            if (winner == None):
                winner = i
            elif i.fitness < winner.fitness:
                winner = i

        return winner

    def population_init(self):
        self.population = []
        for i in range(self.population_size):
            newChromosome = ImgChromo(self.width, self.length)
            # sets newChromosome.fitness
            newChromosome.compute_fitness(self.target_image)
            self.population.append(newChromosome)
        print("fitness array", self.evaluate())
        return self.evaluate()

    def evaluate(self):
        print(f"fitness_values: {self.population[3].fitness}")
        print("Evaluating")
        fitness_array = np.array([chromosome.compute_fitness(
            self.target_image) for chromosome in self.population])
        print(f"fitness_values: {fitness_array}")
        return fitness_array

    def evolutionary_process(self, generations):
        '''
        Essentially overwrites the generation function in EA_base, however, with a few key differences.
        A mix of the code from our implementation of EA in EA_base.py and SebastianCharmot's in 
        https://medium.com/@sebastian.charmot/genetic-algorithm-for-image-recreation-4ca546454aaa

        Fixes: this may not be removing children from the population

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

        # list of fitness values
        # fitness_list = self.population_init()
        # fitness_values = np.sum(fitness_list) - fitness_list

        for i in range(self.num_generations):  # can this be replaced with EA.Generation?
            new_pop = []  # new children to be added here
            fittest_estimate = float('inf')
            # why not replace with population_size and a simple int counter, since len operations take longer
            while len(new_pop) < len(self.population):
                parent_one = self.tournament_select(self.population)
                parent_two = self.tournament_select(self.population)

                fittest_estimate = min(
                    parent_one.fitness, parent_two.fitness, fittest_estimate)

                sheSaidYes = random.uniform(0, 1)

                if sheSaidYes < 0.3:
                    child = self.crossover_random(parent_one, parent_two)

                    while child == None:
                        parent_one = self.tournament_select(self.population)
                        parent_two = self.tournament_select(self.population)

                        child = self.crossover(parent_one, parent_two)

                # elif sheSaidYes <= 0.9:
                #     child = self.crossover_2(parent_one, parent_two, 0.5)

                #     while child == None:
                #         parent_one = self.tournament_select(self.population)
                #         parent_two = self.tournament_select(self.population)

                #         child = self.crossover_2(parent_one, parent_two, 0.5)

                else:
                    child = self.mutate_pixels(parent_one)

                    while child == None:
                        pparent_one = self.tournament_select(self.population)
                        parent_two = self.tournament_select(self.population)
                        child = self.mutate_pixels(parent_one)

                # accept the child into new population list
                new_pop.append(child)
            # add the new children to the population
            self.population = new_pop

            # fitness params recording
            if i % 100 == 0 or i == generations - 1:
                params['epoch'].append(i)
                params['fitness_estimate'].append(fittest_estimate)
                params['crossover_used'].append("crossover_1")
                params['pop_gen_used'].append("random_image_array_1")
                params['im_size'].append(
                    "(" + str(self.width) + "," + str(self.length) + ")")

            # book-keeping
            if i % 100 == 0 or i == generations - 1:

                print(datetime.datetime.now().strftime("%H:%M:%S"), "\tMost fit individual in generation " + str(i) +
                      " has fitness: " + str(fittest_estimate))

                self.population.sort(key=lambda ind: ind.fitness)
                fittest = self.population[0]

                fittest.image.save("gif/fittest_cropped_mona_" + str(i)+".png")

                data_df = DataFrame(params)

                data_df.to_csv("data_cross.csv")

        # save collected data to csv
        data_df = DataFrame(params)
        data_df.to_csv("data_cross.csv")

        # fittest individual of the final population
        self.population.sort(key=lambda ind: ind.fitness)
        fittest = self.population[0]

        return fittest

    def crossover_random(self, parent1, parent2):
        """
        Basic Multiply crossover selects random pixels between each parent and 
        creates a new child multiplying those pixel values.

        Returns the child only if the child is fitter than both parents
        """
        first = np.random.randint(
            2, size=(self.width, self.length, 4))

        second = 1 - first

        first_half_child = np.multiply(first, parent1.img_array)
        print(f'first: {first}\n parent_one: {parent1}')
        second_half_child = np.multiply(second, parent2.img_array)

        child_array = np.add(first_half_child, second_half_child)

        newChild = ImgChromo(self.img_size[1], self.img_size[0])

        newChild.image = Image.fromarray(child_array.astype(np.uint8))
        newChild.img_array = child_array.astype(np.uint8)

        newChild.getfitness(self.target_image)

    def crossover(self, parent1, parent2):
        """
        Inserts random parts of parent1 chromosome into parent2 - just like crossover in the EA class.
        Returns the child only if the child is fitter than both parents
        """
        new_child = super().crossover(parent1, parent2)
        if new_child.compute_fitness(self.target_image) > parent1.compute_fitness(self.target_image) and new_child.compute_fitness(self.target_image) > parent2.compute_fitness(self.target_image):
            return new_child
        else:
            return None  # different from Mustafa's implementation, but more deterministic and less explorative crossover
            # return new_child  # debugging

    def crossover_2(self, ind1, ind2, horizontal_prob):
        """
        Performs 'crossover point' crossover given two parents and creates a child \
            Randomly selects the crossover point to be either a row or column \
            Everything up until the crossover point is from parent 1, everything after is parent 2

        Keyword arguments:
        ind1 -- parent number 1
        ind2 -- parent number 2

        Returns:
        child or None -- child of the two parents if it is more fit than both parents

        Adapted from https://github.com/SebastianCharmot/Genetic-Algorithm-Image-Recreation/blob/master/GP.py
        """

        rand = random.random()

        # perform horizontal crossover point
        if rand <= horizontal_prob:

            split_point = random.randint(1, self.width)

            first = np.ones((split_point, self.length))
            first = np.vstack(
                (first, np.zeros((self.width-split_point, self.length))))

        # perform vertical crossover point
        else:
            split_point = random.randint(1, self.length)

            first = np.ones((self.width, split_point))

            first = np.hstack(
                (first, np.zeros((self.width, self.length-split_point))))

        second = 1 - first
        # Creates the 4 dimensional versions to perform the mutliplying across all color channels
        first = np.dstack([first, first, first, first])
        second = np.dstack([second, second, second, second])

        # Multiply parent1 with first and multiply parent2 with second. Then simplay add them element wise and it should produce the crossover child.

        half_chromo_1 = np.multiply(first, ind1.img_array)
        half_chromo_2 = np.multiply(second, ind2.img_array)

        child_array = np.add(half_chromo_1, half_chromo_2)

        child = ImgChromo(self.length, self.width)

        child.image = Image.fromarray(child_array.astype(np.uint8))
        child.img_array = child_array.astype(np.uint8)

        child.calculate_fitness(self.target_image)

        # elitism
        if child.fitness == min(ind1.fitness, ind2.fitness, child.fitness):
            return child

        return None

    def mutate_pixels(self, imgChoromo_, pixels=40):
        """
        Mutates by updating random pixels; a bit too explorative. 
        Maybe try more epxloitative approach by only choosing pixels if they increase fitness. 
        Could also use list of elites to adopt pixels from.

        """
        # assign random values to
        for i in range(pixels):
            x = random.randint(0, self.length-1)
            y = random.randint(0, self.width-1)
            z = random.randint(0, 3)

            imgChoromo_.img_array[x][y][z] = imgChoromo_.img_array[x][y][z] + \
                random.randint(-10, -10)
            # update PIL image
            imgChoromo_.image = Image.fromarray(imgChoromo_.img_array)
            imgChoromo_.compute_fitness(self.target_image)

    def get_fitness(self, chromosome):  # for compatibility
        return ImgChromo.compute_fitness(self.target_image)


def run_evolution():
    gp = EvolutionaryLisa(r'mona.png')
    # population_size = 100
    generations = 500
    fittest = gp.evolutionary_process(500)
    plt.imshow(fittest.image)
    plt.show()


if __name__ == "__main__":
    run_evolution()
