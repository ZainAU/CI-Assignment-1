import numpy as np 
from EA_Base import EA
from Selection import Selection
rng = np.random.default_rng()


class TSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset = "qa194.tsp",
                 mutatation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10
                 ):
        super().__init__(seed, population_size, dataset, mutatation_rate, offspring_number,  num_generations, Iterations )
        population =  self.population_init()
        return 
    def population_init(self):
        '''
        Question to self:
        1. Shouldn't the starting node be fixed in a TSP?
        '''
        self.dataset = self.dataLoader()
        self.population = np.tile(np.arange(self.chromosome_length), [self.population_size,1]) # Not adding one, This will make indexing a little easier 
        # Randomize the population
        self.seed.permuted(self.population, axis = 1, out = self.population)
        # print(self.population)
        return
    def evaluate(self):
        return np.array([self.get_fitness(chromosome) for chromosome in self.population])
    def get_fitness(self,chromosome):
        return np.sum([self.dataset[chromosome[i],chromosome[i+1]] for i in range(self.chromosome_length-1)])
        
    def dataLoader(self):
        lst = []
        with open("qa194.tsp","r") as file:
            lst = [i.strip().split() for i in file]
        self.chromosome_length = len(lst)
        for _ in range(7):
            lst.pop(0)
        lst.pop(-1)
        self.chromosome_length = len(lst)
        lst = [[float(i) for i in j] for j in lst]
        distance = np.zeros([self.chromosome_length,self.chromosome_length])
        for i in range(self.chromosome_length):
            for j in range(self.chromosome_length):
                distance[i,j] = distance[j,i] = self.ManhattanDistance(lst[i][1:], lst[j][1:])
      
        return distance 
    def ManhattanDistance(self, p1,p2):
        return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**0.5
    def mutation(self, chromosome):
        chromosome
        return


if __name__ == '__main__':
    obj = TSP_EA()
    print(obj.get_fitness(obj.population[2]))
    print(obj.selection_scheme.fitness_proportional_sampling( obj.population, obj.evaluate()))