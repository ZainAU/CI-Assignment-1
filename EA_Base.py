import numpy as np 
from Selection import Selection
rng = np.random.default_rng()
class EA:
    def __init__(self, seed = rng, population_size = 30, dataset = "qa194.tsp", 
                 mutation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10, selection_method = 'FPS',mutation_type = 'insert'):
        self.selection_method = selection_method
        self.seed = seed
        self.population_size = population_size
        self.chromosome_length = None
        self.population = None
        self.mutation_rate = mutation_rate
        self.offspring_number = offspring_number
        self.num_generations = num_generations
        self.Iterations = Iterations
        self.selection_scheme = Selection(offspring_number=self.offspring_number, seed=seed)
        self.mutation_type = mutation_type
        return
    
    def load_data(self):
        return 
    def population_init(self):
        return 
    def mutation(self, chromosome):
        i1 = self.seed.choice(np.arange(self.chromosome_length)) # selecting random indexes to insert or swap
        i2 = self.seed.choice(np.arange(self.chromosome_length))
        
        if self.mutation_type == 'insert':  
            chromosome = list(chromosome)
            node = chromosome.pop(i1)
            # print('Node at = ',node, chromosome[i2])
            chromosome.insert(i2, node)
            chromosome = np.array(chromosome)
        elif self.mutation_type == 'swap':
            chromosome[[i1,i2]] = chromosome[[i2,i1]] 

        return chromosome
    def evaluate(self):
        return np.array([self.get_fitness(chromosome) for chromosome in self.population])
    def get_fitness(self,chromosome):
        return
    def selection(self):
        pass
    def crossover(self, parent1, parent2):
        i1 = self.seed.choice(np.arange(self.chromosome_length))
        i2 = self.seed.choice(np.arange(self.chromosome_length))
        child = parent1.copy()
        child[0:i1] = parent2[0:i1]
        child[i2:] = parent2[i2:] 
        return np.array(child)
    def Generation(self):
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
    def main(self):
        pass