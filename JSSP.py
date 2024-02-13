import numpy as np 
from EA_Base import EA
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class JSSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset="qa194.tsp", mutation_rate=0.5, offspring_number=10, num_generations=50, Iterations=10, selection_method='FPS', mutation_type='insert', optimization_type='minimization',path = "abz5.txt"):
        super().__init__(seed= rng,population_size=population_size, 
                         dataset=dataset, mutation_rate=mutation_rate, offspring_number=offspring_number, num_generations=num_generations, Iterations=Iterations, selection_method=selection_method, mutation_type=mutation_type, optimization_type=optimization_type)
        self.path = path
        self.population_init()
        return 
    
    def population_init(self):
        self.dataLoader()
        np.array([(1,2),(3,4)], dtype="i,i") 
        
        self.population = np.zeros([self.population_size,self.chromosome_length,2],dtype = int)
        ## Gives a numpy array of tuples, these tuples will store the operation information i.e. O_{ij} = ith job and jth operation

        ######### Initializing by permuting the colum of the job sequence matrix and then putting that into the chromosome, i.e. ensuring i-th operation for a job will always come before the j-th for all i < j
        self.J = np.shape(self.job_sequence_matrix)[0]
        self.M = np.shape(self.job_sequence_matrix)[1]
        for chrom_index in range(self.population_size):
            index = 0
            for j in range(0, J*M,J): #jth operation
                
                permuted_index = self.seed.permuted(np.arange(J))
                
                self.population[chrom_index,j:j+J,0] = permuted_index # filling the job index first
                self.population[chrom_index,j:j+J,1] = np.ones(J)*index 
                index +=1
        
        return 
    def dataLoader(self):
        with open(self.path, "r") as file:
            lst = [[int(j) for j in i.strip().split()] for i in file]
            J = lst[0][0]
            M = lst[0][1]
            ############ Setting the chromosome length ################3
            self.chromosome_length = J*M
            ''' The dataset has an implicit assumtion that each machine will be used exactly once in a job, i.e. the number of processes in a job equals to the number of machines'''
            self.job_sequence_matrix = np.zeros([J, M])
            self.process_time_matirx = np.zeros([J,M])


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
        machine_in_use = np.zeros([1,self.M], dtype = bool)
        conc_block = list()
        c_index = [0,0]
        for i in len(chromosome):

            j,o = chromosome[i,:]
            m = self.job_sequence_matrix[j,o]
            if not machine_in_use[m]:
                if 
                # time += self.process_time_matirx[i,m]
                print(j)
        
        return time
    def evaluate(self):
        return super().evaluate()

if __name__ == '__main__':
    obj = JSSP_EA()
    
    print(obj.evaluate())
    # print(x)
