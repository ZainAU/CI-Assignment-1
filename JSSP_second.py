import numpy as np 
from EA_Base import EA
from Selection import Selection
import matplotlib.pyplot as plt
rng = np.random.default_rng()
mutation_type = 'insert'

class JSSP_EA(EA):
    def __init__(self, seed=rng, population_size=30, dataset="qa194.tsp", mutation_rate=0.5, offspring_number=10, num_generations=50, Iterations=10, parent_selection='FPS',survival_selection ='Trunc', mutation_type='insert', optimization_type='minimization',path = "abz5.txt"):
        super().__init__(seed= seed,population_size=population_size, 
                         dataset=dataset, mutation_rate=mutation_rate, offspring_number=offspring_number, num_generations=num_generations, Iterations=Iterations, parent_selection_method=parent_selection,survival_selection=survival_selection, mutation_type=mutation_type, optimization_type=optimization_type)
        self.path = path
        self.population_init()
        return 
    
    def crossover(self, parent1, parent2):
   
        i1 = self.seed.choice(np.arange(self.chromosome_length//2))
        i2 = int(i1+self.chromosome_length//2)
 
        offspring = np.ones(np.shape(parent1),dtype=int) * (self.J+1) #multiplying so that np.where doesnt give wrong answer for job == 0
        parent1attribute = parent1[i1:i2]
        offspring[i1:i2]  =  parent1attribute
        j = 0
        i = 0
        for job in parent2:
            indices = np.where(offspring == job)[0] 
            if len(indices) < self.J:
                if i == i1:
                    i = i2
                offspring[i] = job     
                i += 1 
            if i == self.chromosome_length:
                break
        return offspring
    
    def population_init(self):
        self.dataLoader()
        self.J = np.shape(self.job_sequence_matrix)[0]
        self.M = np.shape(self.job_sequence_matrix)[1]
        
        self.population = self.seed.permuted(np.tile(np.tile(np.arange(self.J), [self.M]),[self.population_size,1]),axis = 1)
        # self.population = np.zeros([self.population_size,self.chromosome_length],dtype = int)
        ## Gives a numpy array of tuples, these tuples will store the operation information i.e. O_{ij} = ith job and jth operation
        # print(self.population)
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
            print(np.shape(self.job_sequence_matrix))


            for i in range(1,len(lst)):
                k = 0
                for j in range(0,len(lst[i]),2):
                    self.job_sequence_matrix[i-1,k] = lst[i][j]
                    self.process_time_matirx[i-1,k] = lst[i][j+1]
                    k+=1
               
        return 
    def get_order(self, chromosome):
        order = np.zeros(len(chromosome),dtype=int)
        for i in range(self.J):
            job_index = np.where(chromosome == i )
            order_ind = 0
            for j in job_index[0]:
                order[j] = order_ind 
                order_ind += 1
        return order
    def mutation(self, chromosome):
        return super().mutation(chromosome)
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
        # print(order)
        # print(chromosome)
        for i in range(self.chromosome_length):
            j = chromosome[i]
            o = order[i]
   
            # print(j,o, end = '  ')
            # print(self.job_sequence_matrix)
            try:
                m = self.job_sequence_matrix[j,o]
            except:
                print(i)
                print(j, o)
                print(order)
                print(self.job_sequence_matrix)
                print(chromosome)
                sda
            # print(previous_job[j])
            
            maxi = np.max([current_machine[m], previous_job[j]]) + self.process_time_matirx[j,o]
            current_machine[m] = maxi
            previous_job[j] = maxi
        time = np.max(previous_job)
        return time
    def evaluate(self):
        return super().evaluate()
    def Generation(self):
        return super().Generation()
    def main(self,selection_methods):
        '''ssssssssssssss'''
        
        for selection in selection_methods:
            Best = np.inf
            self.parent_selection_method = selection[0]
            self.survival_Selection_method = selection[1]
         
            average_best_fitness = np.zeros([self.Iterations, self.num_generations])
            avg_avg_fitness = np.zeros([self.Iterations, self.num_generations])
            for i in range(self.Iterations):
                
                best_fit,average_fit= super().main()
                print(f'Best of iteration-{i} = {np.min(best_fit)}')
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
            fig.savefig(f'JSSP_figs/{self.path[:-4]}-Best-fit-{int(Best)}-parent-{self.parent_selection_method}-survival-{self.survival_Selection_method}-{self.population_size}-{self.mutation_rate}-iteration-{i}.png')
            # plt.show()

    
if __name__ == '__main__':
    seed = np.random.default_rng(4)
    mutation_rate = 0.5
    num_generations = 1000
    suvivor_Selection = 'BT'
    parent_Selection = 'Trunc'
    optimization_type='minimization'
    population_size = 100
    offspring_number = 60
    iterations = 1
    path = 'abz7.txt'
    
    obj = JSSP_EA(num_generations=num_generations,
                optimization_type=optimization_type,
                survival_selection=suvivor_Selection,
                population_size=population_size,
                parent_selection=parent_Selection,
                Iterations= iterations,
                mutation_rate=mutation_rate,
                seed=seed,
                offspring_number=offspring_number,
                path = path)
    selection_criteria = [('FPS','Random'),('BT', 'Trunc'),('Trunc','Trunc'),('Random','Random'),('FPS', 'RBS')]
    obj.main(selection_methods=selection_criteria)

    # obj.get_order(obj.population[1,:])

    # print(obj.population[-1,:])
    # print(np.shape(obj.population[-1,:]))
    # print(obj.crossover(obj.population[-1,:],obj.population[-2,:]))
    # print(obj.mutation(obj.population[-1,:]))
    # # print(x)
