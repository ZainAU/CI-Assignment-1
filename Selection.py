import numpy as np 
rng = np.random.default_rng()

class Selection:
    def __init__(self,offspring_number, seed = rng ):
        self.seed = seed
        self.offspring_number = offspring_number
        self.selection_function = {'FPS':self.fitness_proportional_sampling,
                                   'Trunc': self.truncation,
                                   'RBS':self.rank_based_selection,
                                   'Random':self.random_selection,
                                   'BT': self.binary_tournament}
    def fitness_proportional_sampling(self, population, fitness_list, num_samples):
        samples = list()

        norm_fitness = np.array(fitness_list)
        norm_fitness = norm_fitness/np.sum(norm_fitness)    
        # print(population)    
        c = 0
        cdf = []
        for i in range(len(norm_fitness)):
            c += norm_fitness[i]
            cdf.append(c)
        for i in range(num_samples):
            probablity = self.seed.random()    
            for j in range(len(cdf)):
                if probablity <= cdf[j]:
                    samples.append(population[j,:]) 
                    break

        samples = np.array(samples)   
        # print(f'len of samples {len(samples)}')
        '''# print(f'The number of parents are {len(parents)}, number of children {self.offspring_number}')
        # print(len(cdf))
        # print(len(parents))'''
        return samples
    def truncation(self,population, fitness_list, num_samples):
        return population[np.argsort(fitness_list),:][-num_samples:]
    def rank_based_selection(self,population, fitness_list, num_samples):
        sorted_indices = np.argsort(fitness_list)[::-1] #This line gives the indices of our population sorted in descending order
        sorted_population = population[sorted_indices]
        ranks = np.arange(len(population))[::-1]+1 # ranks on the newly sorted population
        samples = self.fitness_proportional_sampling(population=sorted_population,fitness_list=ranks,num_samples=num_samples)
        return samples
    def binary_tournament(self,population, fitness_list, num_samples):
        samples = []
        return samples
    def random_selection(self, population,num_samples, fitness_list = None):
        return self.seed.choice(population, num_samples)

    def get_parents(self, population,fitness_list, func = 'FTP'):
        num_samples = self.offspring_number*2
        return self.selection_function[func](population,fitness_list, num_samples)
    
    def get_survivor(self,population, fitness_list, num_survivors, func = 'FTP' ):
        return self.selection_function[func](population,fitness_list, num_survivors)
    def stochastic_universal_sampling(self):
        pass
    