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
        average_fit = super().main()
        

        plt.plot(average_fit)
        plt.ylabel("Total distances")
        plt.xlabel("100th Generations")
        # plt.text(0.45,0.5, "Selection method")
        plt.title(f"Populatation size ={self.population_size}, selection scheme = {self.selection_method}")
        plt.show()
        print(f'Final value = {average_fit[-1]}')




if __name__ == '__main__':
    mutation_rate = 0.25
    num_generations = 1000
    slection_method = 'RBS'
    optimization_type='minimization'
    population_size = 1000
    obj = TSP_EA(num_generations=num_generations,
                optimization_type=optimization_type,
                selection_method=slection_method,
                population_size=population_size,
                mutation_rate=mutation_rate)
    # parent1 = obj.population[1,:]
    # parent2 = obj.population[2,:]
    # child = obj.crossover(obj.population[1,:],obj.population[2,:])
    # print(f'parent1 = {parent1} \nparent2 = {parent2}')
    # print(f'CHild - {child}')

    obj.main()


    
