import numpy as np 
from EA_Base import EA
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class TSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset = "qa194.tsp",
                 mutation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10, selection_method = 'FPS', 
                 mutation_type = mutation_type, optimization_type = 'minimization'
                 ):
        super().__init__(seed, population_size, dataset, mutation_rate, offspring_number,  num_generations, Iterations, selection_method,mutation_type , optimization_type)
        population =  self.population_init()
        return 
    def population_init(self):
        self.dataset = self.dataLoader()
        self.population = np.tile(np.arange(self.chromosome_length), [self.population_size,1]) # Not adding one, This will make indexing a little easier 
        # Randomize the population
        self.seed.permuted(self.population, axis = 1, out = self.population)
        # print(self.population)
        return
    
    def evaluate(self):
        return super().evaluate()
        # return np.array([self.get_fitness(chromosome) for chromosome in self.population])
    
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
        return super().mutation(chromosome)
    

    def crossover(self, parent1, parent2):
        return super().crossover(parent1, parent2)
    
        
    def Generation(self):
        return super().Generation()
    
    def main(self):
        best_fit,average_fit = super().main()
        
        plt.subplot(1,2,1)
        plt.plot(average_fit)
        plt.ylabel("Total distances")
        plt.xlabel("100th Generations")
        plt.title(f"selection scheme = {self.selection_method}, Average fit")
        plt.subplot(1,2,2)
        plt.plot(best_fit)
        plt.ylabel("Total distances")
        plt.xlabel("100th Generations")
        # plt.text(0.45,0.5, "Selection method")
        plt.title(f"Populatation size ={self.population_size}, Best Fit")
        plt.show()
        print(f'Final value = {average_fit[-1]}')




if __name__ == '__main__':
    seed = np.random.default_rng(42)
    mutation_rate = 0.5
    num_generations = 50000
    slection_method = 'BT'
    optimization_type='minimization'
    population_size = 30
    offspring_number = 10
    obj = TSP_EA(num_generations=num_generations,
                optimization_type=optimization_type,
                selection_method=slection_method,
                population_size=population_size,
                mutation_rate=mutation_rate,
                seed=seed,
                offspring_number=offspring_number)

    obj.main()


    
