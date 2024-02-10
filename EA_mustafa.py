import numpy as np 
from EA_Base import EA
from Selection import Selection
rng = np.random.default_rng()


class Selection:
    def __init__(self,offspring_number, seed = rng ):
        self.seed = seed
        self.offspring_number = offspring_number
    def fitness_proportional_sampling(self, population, fitness_list):
        parents = list()
        norm_fitness = np.array(fitness_list)
        norm_fitness = norm_fitness/np.sum(norm_fitness)    
        print(population)    
        c = 0
        cdf = []
        for i in range(len(norm_fitness)):
            c += norm_fitness[i]
            cdf.append(c)
        for i in range(self.offspring_number*2):
            probablity = self.seed.random()    
            for j in range(len(cdf)):
                if probablity <= cdf[j]:
                    parents.append(population[j,:]) 
                    break

        parents = np.array(parents)   
            
        # print(f'The number of parents are {len(parents)}, number of children {self.offspring_number}')
        # print(len(cdf))
        # print(len(parents))

        return parents
    def stochastic_universal_sampling(self):
        pass
    
    

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