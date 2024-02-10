import numpy as np 
rng = np.random.default_rng()
class EA:
    def __init__(self, seed = rng, population_size = 30, dataset = "qa194.tsp", 
                 mutatation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10, selection_method = 'FPS'):
        self.selection_method = selection_method
        self.seed = seed
        self.population_size = population_size
        self.chromosome_length = None
        self.population = None
        self.mutatation_rate = mutatation_rate
        self.offspring_number = offspring_number
        self.num_generations = num_generations
        self.Iterations = Iterations
        self.selection_scheme = Selection(offspring_number=self.offspring_number, seed=seed)
        return
    
    def load_data(self):
        return 
    def population_init(self):
        return 
    def mutation(self, chromosome):
        return
    def get_fitness(self,chromosome):
        return
    def selection(self):
        pass
    def crossover(self):
        pass
    def evaluate(self):
        return
    def main(self):
        pass