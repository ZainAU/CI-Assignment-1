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


rng = np.random.default_rng()

class Selection:
    def __init__(self, type = 'FTP',seed = rng):

        pass
    


class EA:
    def __init__(self, seed = rng, population_size = 100, dataset = "qa194.tsp" ):
        selection_scheme = Selection()
        self.seed = seed
        self.population_size = population_size
        self.chromosome_length = None
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
    def __init__(self, seed=rng, population_size=100, dataset = "qa194.tsp" ):
        super().__init__(seed, population_size, dataset = "qa194.tsp" )
        population =  self.population_init()
        return 
    def population_init(self):
        self.dataset = self.dataLoader()
        self.population = np.tile(np.arange(self.chromosome_length), [self.population_size,1]) + 1
        self.seed.permuted(self.population, axis = 1, out = self.population)
        print(self.population)
        
        return
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
         

if __name__ == '__main__':
    obj = TSP_EA()
    print("Sued")