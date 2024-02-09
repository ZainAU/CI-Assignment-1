import numpy as np 
from abc import ABC, abstractmethod 

'''
    Note to self:
        Look up what an abstract base class is?
    Maybe make a class to handle the population as well?
    - Assumption is that the the data, i.e. the graphs exists as a adjacency Matrix
'''
def load_Dataset():
    
    return


rng = np.random.default_rng(seed=42)

class Selection:
    def __init__(self, type = 'FTP',seed = rng ):

        pass
    


class EA:
    def __init__(self, seed = rng, population_size = 100, chromosome_length= 10):
        selection_scheme = Selection()
        self.seed = seed
        self.population_size = population_size
        self.chromosome_length = chromosome_length 
        self.population = None
        return
    
    def load_data(self):
        return 
    def population_init(self):
        return 
    def mutation(self, parent):
        
        pass
    def selection(self):
        pass
    def crossover(self):
        pass
    def main(self):
        pass
class TSP_EA(EA):
    def __init__(self, seed=rng, population_size=100, chromosome_length=10):
        super().__init__(seed, population_size, chromosome_length)
        population =  self.population_init()
        return 
    def population_init(self):
        self.population = self.seed

if __name__ == '__main__':
    pass