import numpy as np 
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
    