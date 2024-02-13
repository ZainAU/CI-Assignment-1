import numpy as np 
from EA_Base import EA
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class JSSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset="qa194.tsp", mutation_rate=0.5, offspring_number=10, num_generations=50, Iterations=10, parent_selection='FPS',survival_Selection ='Trunc', mutation_type='insert', optimization_type='minimization',path = "abz5.txt"):
        super().__init__(seed= seed,population_size=population_size, 
                         dataset=dataset, mutation_rate=mutation_rate, offspring_number=offspring_number, num_generations=num_generations, Iterations=Iterations, parent_selection_method=parent_selection,survival_selection=survival_Selection, mutation_type=mutation_type, optimization_type=optimization_type)
        self.path = path
        self.population_init()
        return 
    def get_order(self, chromosome):
        order = np.zeros(len(chromosome),dtype=int)
        for i in range(self.J):
            job_index = np.where(chromosome == i )
            order_ind = 0
            for j in job_index[0]:
                order[j] = order_ind 
                order_ind += 1
                # print(order)
        return order
    def crossover(self, parent1, parent2):
        i1 = self.seed.choice(np.arange(self.chromosome_length//2))
        i2 = int(i1+self.chromosome_length//2)
        offspring = parent1.copy()
        parent1_attribute = parent1[i1:i2]
        parent2_attribute =[]
        ''''''
        return offspring
  
    def population_init(self):
        self.dataLoader()
        self.J = np.shape(self.job_sequence_matrix)[0]
        self.M = np.shape(self.job_sequence_matrix)[1]
        self.population = self.seed.permuted(np.tile(np.tile(np.arange(self.J), [self.M]),[self.population_size,1]))
        # self.population = np.zeros([self.population_size,self.chromosome_length],dtype = int)
        ## Gives a numpy array of tuples, these tuples will store the operation information i.e. O_{ij} = ith job and jth operation
        print(self.population)
        return 
    def dataLoader(self):
        with open(self.path, "r") as file:
            lst = [[int(j) for j in i.strip().split()] for i in file]
            self.J = lst[0][0] # number of jobs
            self.M = lst[0][1] #Number of machines
            ############ Setting the chromosome length ################3
            self.chromosome_length = self.J*self.M
            ''' The dataset has an implicit assumtion that each machine will be used exactly once in a job, i.e. the number of processes in a job equals to the number of machines'''
            self.job_sequence_matrix = np.zeros([self.J, self.M],dtype= int)
            self.process_time_matirx = np.zeros([self.J,self.M],dtype =int)
            


            for i in range(1,len(lst)):
                k = 0
                for j in range(0,len(lst[i]),2):
                    self.job_sequence_matrix[i-1,k] = lst[i][j]
                    self.process_time_matirx[i-1,k] = lst[i][j+1]
                    k+=1
               
        return 
    
    
    def get_fitness(self, chromosome):
        '''
        Assumption is that the chromosome is valid: so dont need to verify the sequence constraint
        Use concurrent time machine use time heuristic. This would have two conditions 
        1. Machine is free
        2. Different machines
        concurrent block is represented by [startindex, end index]

        '''
        time = 0
        order = self.get_order(chromosome)

        current_machine = np.zeros(self.M,dtype = int)
        previous_job = np.zeros(self.J,dtype = int)
        for i in range(self.chromosome_length):
            j = chromosome[i]
            o = order[i]
            m = self.job_sequence_matrix[j,o]
            print(previous_job[j])
            
            maxi = np.max(current_machine[m], previous_job[j]) + self.process_time_matirx[j,o]
            current_machine[m] = maxi
            previous_job[j] = maxi
        time = np.max(previous_job)
        return time
    def evaluate(self):
        return super().evaluate()

if __name__ == '__main__':
    obj = JSSP_EA()
    obj.get_order(obj.population[1,:])
    print(obj.evaluate()) 
    # print(x)
