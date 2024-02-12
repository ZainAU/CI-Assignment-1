import numpy as np 
from EA_Base import EA, rng
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class JSSP_EA(EA):
    def __init__(self, seed=..., population_size=30, dataset="qa194.tsp", mutation_rate=0.5, offspring_number=10, num_generations=50, Iterations=10, selection_method='FPS', mutation_type='insert', optimization_type='minimization',path = "abz5.txt"):
        super().__init__(seed, population_size, dataset, mutation_rate, offspring_number, num_generations, Iterations, selection_method, mutation_type, optimization_type)
        self.path = path
        population =  self.population_init()
        return 
    
    def population_init(self):
        
        return 
    def dataLoader(self):
        Jobs = dict()
        with open(self.path, "r") as file:
            lst = [[int(j) for j in i.strip().split()] for i in file]
            J = lst[0][0]
            M = lst[0][1]

            for i in range(1,len(lst)):
                Jobs[i-1] = list()
                for j in range(0,len(lst[i]),2):
                    Jobs[i-1].append((lst[i][j], lst[i][j+1]))
        return Jobs
    
    def get_fitness(self, chromosome):
        time = 0 
        for key,value in chromosome.items():
            for _,_,start,end in value:
                time += end-start
        return time
    def evaluate(self):
        return super().evaluate()

if __name__ == '__main__':
    obj = JSSP_EA()
    x = obj.dataLoader()
    # print(x)
    chromosome = {0:[(1,2,8,15)],
                  1:[(1,2,8,10),(1,2,8,10),(1,2,8,10)]}
    print(obj.get_fitness(chromosome))
