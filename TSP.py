import numpy as np 
from EA_Base import EA
from Selection import Selection
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
    # def Generation(self):
        fitness_list = self.evaluate()
        parents = self.selection_scheme.get_parents(self.population,fitness_list, self.selection_method)
        offspring = list()
        # Perform recombination
        for i in range(0,len(parents),2):
            offspring.append(self.crossover(parents[i,:],parents[i+1,:]))
        # print(f'The number of offsprings are: {len(offspring)}')
        offspring = np.array(offspring)
        # print(len(self.population))
        self.population = np.concatenate((self.population,offspring), axis = 0 )
        # print(len(self.population))
        # perform mutation
        ''''
        Is it okay to mutate the population after recombination?
        '''
        mutation_prob = self.seed.random([len(self.population)])
        mutation_mask = (mutation_prob <= self.mutation_rate)
        mask_indices = np.where(mutation_mask == True)[0]
        # print(f'Mutation inddices = {mask_indices}')
        # print(self.population)
        # print("The length of the mask")
        # print(self.population[mutation_mask,:])
        
        for i in mask_indices:
            # print(f'the indices ={i}')
            self.population[i,:] = self.mutation(self.population[i,:])

        count = len(mask_indices)
        
        print(f"The number of mutations are = {count}")

        # print(offspring)
        return 

if __name__ == '__main__':
    num_generations = 30000
    slection_method = 'FPS'
    optimization_type='minimization'
    obj = TSP_EA(num_generations=num_generations, optimization_type=optimization_type,selection_method=slection_method)
    # parent1 = obj.population[1,:]
    # parent2 = obj.population[2,:]
    # child = obj.crossover(obj.population[1,:],obj.population[2,:])
    # print(f'parent1 = {parent1} \nparent2 = {parent2}')
    # print(f'CHild - {child}')

    obj.main()


    
