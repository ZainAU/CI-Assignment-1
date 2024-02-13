import numpy as np 
from EA_Base import EA
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class TSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset = "qa194.tsp",
                 mutation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10,parent_selection_method = 'FPS',survival_selection = 'Trunc',
                 mutation_type = mutation_type, optimization_type = 'minimization'
                 ):
        super().__init__(seed, population_size, dataset, mutation_rate, offspring_number,  num_generations, Iterations, parent_selection_method ,survival_selection,mutation_type , optimization_type)
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
    
    def main(self, selection_methods):
        for selection in selection_methods:
            np.min(best_fit)
            self.parent_selection_method = selection[0]
            self.survival_Selection_method = selection[1]
            print(self.Iterations)
            average_best_fitness = np.zeros([self.Iterations, self.num_generations])
            avg_avg_fitness = np.zeros([self.Iterations, self.num_generations])
            for i in range(self.Iterations):
                
                best_fit,average_fit= super().main()
                average_best_fitness[i,:] = best_fit
                avg_avg_fitness[i,:] = average_fit
                if np.min(best_fit) < Best:
                    Best = np.min(best_fit)
               
            
            
            
            fig = plt.figure()
            # plt.plot(average_fit,'g')
            plt.plot(np.average(avg_avg_fitness, axis = 0),'b')
            # print(np.average(avg_avg_fitness, axis = 0))
            plt.plot(np.average(average_best_fitness, axis = 0), 'r')


            plt.ylabel("Total distances")
            plt.xlabel("Generations")
            plt.title(f"Average fit and Bestfit")
            plt.legend(['Average fit', 'Best fit'])
     
            # Save the full figure...
            fig.savefig(f'TSP_fig/Best-fit-{int(Best)}-parent-{self.parent_selection_method}-survival-{self.survival_Selection_method}-{self.population_size}-{self.mutation_rate}-iteration-{i}.png')
            # plt.show()
       
                
          




if __name__ == '__main__':
    seed = np.random.default_rng(42)
    mutation_rate = 0.5
    num_generations = 5000
    suvivor_Selection = 'BT'
    parent_Selection = 'Trunc'
    optimization_type='minimization'
    population_size = 100
    offspring_number = 60
    iterations = 4
    obj = TSP_EA(num_generations=num_generations,
                optimization_type=optimization_type,
                survival_selection=suvivor_Selection,
                population_size=population_size,
                parent_selection_method=parent_Selection,
                Iterations= iterations,
                mutation_rate=mutation_rate,
                seed=seed,
                offspring_number=offspring_number)

    selection_criteria = [('FPS','Random'),('BT', 'Trunc'),('Trunc','Trunc'),('Random','Random'),('FPS', 'RBS')]
    obj.main(selection_methods=selection_criteria)


    
