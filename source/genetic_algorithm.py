import numpy as np

class GeneticAlgorithm:
    def __init__(self, latent_dim, population_size, mutation_rate, crossover_rate, fitness_func):
        self.latent_dim = latent_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func  # A function that takes a latent vector and returns a score
        
        # Initialize population with random vectors from a normal distribution
        self.population = [np.random.randn(self.latent_dim) for _ in range(self.population_size)]

    def _crossover(self, parent1, parent2):
        # One-point crossover
        crossover_point = np.random.randint(1, self.latent_dim)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def _mutate(self, individual):
        for i in range(self.latent_dim):
            if np.random.rand() < self.mutation_rate:
                # Add a small amount of noise
                individual[i] += np.random.randn() * 0.01
        return individual

    def evolve(self):
        # 1. Calculate fitness for the entire population
        fitness_scores = [self.fitness_func(ind) for ind in self.population]

        # 2. Select the best individuals to be parents (Elitism + Tournament Selection)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Keep the top 2 best individuals (elitism)
        next_population = [self.population[i] for i in sorted_indices[:2]]

        # 3. Create the rest of the new population through crossover and mutation
        while len(next_population) < self.population_size:
            # Tournament selection: pick 4 random individuals and choose the best 2 as parents
            tournament = np.random.choice(self.population_size, 4, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            parent_indices = np.argsort(tournament_fitness)[::-1][:2]
            parent1 = self.population[tournament[parent_indices[0]]]
            parent2 = self.population[tournament[parent_indices[1]]]

            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy() # Keep one parent if no crossover

            mutated_child = self._mutate(child)
            next_population.append(mutated_child)
        
        self.population = next_population
        
        # Return the best individual and its score from this generation
        best_index = sorted_indices[0]
        return self.population[best_index], fitness_scores[best_index]
