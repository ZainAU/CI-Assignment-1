# decide fitness function
# decide chromosome representation
# initialize the population randomly
#

"""
def compute_fitness(population):
    fitness = {}
    for each member in population:
        fitness[member] = fitness_function(member)
    return fitness
"""

"""
def fitnessProportionalSelection(population, numOfSelected=len(population)): -> alelle
    newPopulation = []
    # newPopulation = ??? 
    # fitnessModel = population.fitness() #returns pre-computed fitness
    fitness = computefitness(population)
    # cumulative_fitness -> dict(alelle: [fitness, cum_fitness])
    for sample in range(len(numOfSelected)):
        total_weight = sum(fitness.values())
        selection_point = random.uniform(0, total_weight)
        
        one approach: 
            compute cumulutive fitness on the go
        another approach: 
            pre-compute cumulative fitness, store it on a list and access it later
        right approach:
        # https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights
        dum dum approach:
        ordered_dictionary = set?

def rankBasedSelection(population, size):
    sortedPopulation = OrderedDict(sorted(fitness.items()))
    selectedPopulation = sortedPopulation[0:size]
    return selectedPopulation

def tournament_selection():
    pass
def random_selection(population, size):
    newPopulation = []
    createcopyofpopluation
    for i in range(size):
        newPopulation.append(population[random.randint(0:len(population))])       
        population.remove(selected)

    return newPopulation


def truncation(population, size):
    pass 
    
def select_parents(populaiton, NumOfSelected = len(population), type="ftinessProportional"):
    if type== "ftinessProportional":
        parents = fitnessProportionalSelection(population, numOfSelected=NumOfSleceted)
    return parents
"""

"""
def createPairs(parents):
    pairings = []
    randomShuffle parents
    for i in range(len(parents)):
        pairings = (parents[i], parents[len(parents)-i])
    return pairings
"""

"""
def crossover(parents, type="order1"):
# types of crosovers: order-1, 
    
"""

"""
def mutation(population, type="swap"):
# types of mutations: random_swap, post_insertion 
"""

# Repeat the follwoing
""" 
    compute_fitness(population)
    parents = select_parents(population) 
    parentPairs = createPairs(parents)
    children = []
    for i in range(len(parentPairs)):
        newChild = crossover(parentPairs, order1)
        children.append(newChild)
        compute_fitness?
    population += children
    compute_fitness(population)?
    toBeTerminated = select_parents(population, size)
    population = population - toBeTerminated
"""

# check if iterations complete
# print fitness levle
