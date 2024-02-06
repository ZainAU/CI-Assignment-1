import random
import sys
from itertools import permutations


def tsp(graph):
    # Number of cities
    n = len(graph)

    # Create a list of all possible permutations of cities
    cities = list(range(n))
    permutations_list = list(permutations(cities))

    # Initialize minimum distance to a large value
    min_distance = sys.maxsize

    # Iterate through all permutations
    for perm in permutations_list:
        # Initialize current distance to 0
        current_distance = 0

        # Check if the current permutation is a valid path
        if perm[0] == 0:
            # Calculate the distance for the current permutation
            for i in range(n - 1):
                current_distance += graph[perm[i]][perm[i + 1]]

            # Add the distance from the last city back to the starting city
            current_distance += graph[perm[n - 1]][perm[0]]

            # Update the minimum distance if the current distance is smaller
            min_distance = min(min_distance, current_distance)

    return min_distance


# Example usage
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

min_distance = tsp(graph)
print("Minimum distance:", min_distance)


def tsp(graph, population_size, num_generations):
    # Number of cities
    n = len(graph)

    # Create a list of all possible permutations of cities
    cities = list(range(n))
    permutations_list = list(permutations(cities))

    # Initialize minimum distance to a large value
    min_distance = sys.maxsize

    # Initialize the population
    population = []
    for _ in range(population_size):
        individual = random.sample(cities, n)
        population.append(individual)

    # Evolutionary loop
    for _ in range(num_generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = []
        for individual in population:
            fitness_scores.append(calculate_distance(graph, individual))

        # Select parents for crossover
        parents = selection(population, fitness_scores)

        # Create offspring through crossover
        offspring = []
        for i in range(population_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            offspring.append(child)

        # Apply mutation to the offspring
        for i in range(population_size):
            offspring[i] = mutation(offspring[i])

        # Replace the population with the offspring
        population = offspring

        # Find the best individual in the population
        best_individual = min(
            population, key=lambda x: calculate_distance(graph, x))
        best_distance = calculate_distance(graph, best_individual)

        # Update the minimum distance if the best distance is smaller
        min_distance = min(min_distance, best_distance)

    return min_distance


def calculate_distance(graph, path):
    # Calculate the distance for the given path
    distance = 0
    n = len(path)
    for i in range(n - 1):
        distance += graph[path[i]][path[i + 1]]
    distance += graph[path[n - 1]][path[0]]
    return distance


def selection(population, fitness_scores):
    # Perform tournament selection to select parents for crossover
    tournament_size = 2
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        best_individual = min(tournament, key=lambda x: fitness_scores[x])
        parents.append(population[best_individual])
    return parents


def crossover(parent1, parent2):
    # Perform ordered crossover to create a child
    n = len(parent1)
    child = [-1] * n
    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)
    child[start:end] = parent1[start:end]
    for i in range(n):
        if parent2[i] not in child:
            for j in range(n):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child


def mutation(individual):
    # Perform swap mutation on the individual
    n = len(individual)
    if random.random() < 0.1:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


# Example usage
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

population_size = 50
num_generations = 100

min_distance = tsp(graph, population_size, num_generations)
print("Minimum distance:", min_distance)


# need to modify
#   - fitness function
#   - Initatialization
