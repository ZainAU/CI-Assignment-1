import numpy as np 
from Selection import Selection
from tqdm import tqdm
import matplotlib.pyplot as plt 
rng = np.random.default_rng()
class EA:
    def __init__(self, seed = rng, population_size = 30, dataset = "qa194.tsp", 
                 mutation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10, selection_method = 'FPS',mutation_type = 'insert',optimization_type = 'minimization'):
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
        self.optimization_type = optimization_type
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
    
    def crossover(self, parent1, parent2):
        i1 = self.seed.choice(np.arange(self.chromosome_length//2))
        i2 = int(i1+self.chromosome_length//2)
        child = parent1.copy()
        parent1_attribute = parent1[i1:i2]
        parent2_attribute =[]
        for value in parent2:
            if not value in parent1_attribute:
                parent2_attribute.append(value)
        j = 0
        for i in range(self.chromosome_length):
            if i >= i1 or i <= i2:
                continue
            else:
                child[i] = parent2_attribute[j]
                j+=1
        assert len(np.unique(child)) == self.chromosome_length
        return np.array(child)
    def Generation(self):
        ########### Compute Fitness ######################
        fitness_list = self.evaluate()                             
        ############## Optimization type ################      
        if self.optimization_type == 'minimization':
            fitness_values = np.sum(fitness_list) - fitness_list 
        elif self.optimization_type == 'maximization':
            fitness_values = fitness_list.copy()
        else:
            AssertionError("Please input correct type of optimization")
        ########### Parent Selection ###################
        parents = self.selection_scheme.get_parents(self.population,fitness_values, self.selection_method)
        offspring = list()
        ##############3 Perform recombination ############
        for i in range(0,len(parents),2):
            offspring.append(self.crossover(parents[i,:],parents[i+1,:]))
        offspring = np.array(offspring)
        ########### perform mutation ##################3
        # print(f'The number of individuals at start of mutation {self.population_size}')
        ''''
        Is it okay to mutate the population after recombination?
        
        Made a mutation mask to make the algorithm more efficient. Now instead of O(N) where N = population size the complexity is O(NP) wehre NP is the number of chromosomes to be mutated
        '''
        mutation_prob = self.seed.random([len(self.population)])
        mutation_mask = (mutation_prob <= self.mutation_rate)
        mask_indices = np.where(mutation_mask == True)[0]

        for i in mask_indices:
            self.population[i,:] = self.mutation(self.population[i,:])

        num_mutations = len(mask_indices)
        ###### Concatenate the offsprings with the parents
        self.population = np.concatenate((self.population,offspring), axis = 0 )

        ############# Perform survivor slection #############
        fitness_list = self.evaluate()
        if self.optimization_type == 'minimization':
            fitness_values = np.sum(fitness_list) - fitness_list 
        elif self.optimization_type == 'maximization':
            fitness_values = fitness_list.copy()
        else:
            AssertionError("Please input correct type of optimization")
        self.population = self.selection_scheme.get_survivor(self.population,fitness_values, self.population_size, self.selection_method)
        return np.min(fitness_list)
    def main(self):
        average_fit = list()
        for i in tqdm(range(self.num_generations)):
            if i%100:
                average_fit.append(self.Generation())
        return average_fit
        
        