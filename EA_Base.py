import numpy as np 
from Selection import Selection
from tqdm import tqdm
import matplotlib.pyplot as plt 
rng = np.random.default_rng()
class EA:
    def __init__(self, seed = rng, population_size = 30, dataset = "qa194.tsp", 
                 mutation_rate = 0.5, offspring_number = 10,  num_generations = 50, Iterations = 10, parent_selection_method = 'FPS',survival_selection = 'Trunc',mutation_type = 'insert',optimization_type = 'minimization'):
        self.parent_selection_method = parent_selection_method
        self.survival_Selection_method  = survival_selection
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
        self.best_chromosome = [None, None]#[chromosome, fitness]
        
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
        best_fitness = self.best_chromosome[1]
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
        parents = self.selection_scheme.get_parents(self.population,fitness_values, self.parent_selection_method)
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
        mutation_prob = self.seed.random([len(offspring)])
        mutation_mask = (mutation_prob <= self.mutation_rate)
        mask_indices = np.where(mutation_mask == True)[0]

        for i in mask_indices:
            offspring[i,:] = self.mutation(offspring[i,:])

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
        self.population = self.selection_scheme.get_survivor(self.population,fitness_values, self.population_size, self.survival_Selection_method)
        
        if self.optimization_type == 'minimization':
            best_so_far = np.min(fitness_list)
            for i in range(len(fitness_list)):
                try:
                    if fitness_list[i] <= best_fitness:
                        self.best_chromosome[1] = fitness_list
                        self.best_chromosome[0] = self.population[i]
                        

                except:
                    self.best_chromosome[1] = fitness_list[i]
        elif self.optimization_type == 'maximization':
            best_so_far = np.max(fitness_list)
            for i in range(len(fitness_list)):
                try:
                    if fitness_list[i] >= best_fitness:
                        self.best_chromosome[1] = fitness_list
                        self.best_chromosome[0] = self.population[i]
                except:
                    self.best_chromosome[1] = fitness_list[i]
                    
        else:
            AssertionError("Please input correct optimization")
        

        return best_so_far,np.average(fitness_list)
    def main(self):
        best_fit_per_generation = list()
        average_fit_per_generation = list()
        average_fit = list()
        best_fit = list()
        for i in tqdm(range(self.num_generations)):          
            best_so_far, average_so_far = self.Generation()
            average_fit.append(average_so_far)
            best_fit.append(best_so_far)
            best_fit_per_generation.append(best_so_far)
            average_fit_per_generation.append(best_so_far)
                
        return best_fit,average_fit
        