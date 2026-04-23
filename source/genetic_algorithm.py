import numpy as np

class GeneticAlgorithm:
    """
    Implémentation d'un algorithme génétique pour optimiser des vecteurs latents
    en fonction d'une fonction de fitness donnée (score).
    """

    def __init__(self, latent_dim, population_size, mutation_rate, crossover_rate, fitness_func):
        """
        Initialise l'algorithme génétique avec les paramètres spécifiés.

        Paramètres:
            latent_dim (int): La dimension du vecteur latent.
            population_size (int): Le nombre d'individus dans la population.
            mutation_rate (float): Le taux de mutation (probabilité de mutation).
            crossover_rate (float): Le taux de croisement (probabilité de croisement).
            fitness_func (callable): Une fonction qui prend un vecteur latent et renvoie un score (fitness).
        """
        self.latent_dim = latent_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func  # Une fonction qui prend un vecteur latent et renvoie un score
        
        # Initialise la population avec des vecteurs aléatoires tirés d'une distribution normale
        self.population = [np.random.randn(self.latent_dim) for _ in range(self.population_size)]

    def _crossover(self, parent1, parent2):
        """
        Effectue un croisement en un point (one-point crossover) entre deux parents
        pour générer un nouvel enfant (vecteur latent).

        Paramètres:
            parent1 (numpy.ndarray): Vecteur latent du premier parent.
            parent2 (numpy.ndarray): Vecteur latent du second parent.

        Retour:
            child (numpy.ndarray): Nouvel individu issu du croisement.
        """
        # Croisement en un point
        crossover_point = np.random.randint(1, self.latent_dim)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def _mutate(self, individual):
        """
        Applique une mutation à un individu en ajoutant aléatoirement du bruit 
        à certaines composantes de son vecteur latent.

        Paramètres:
            individual (numpy.ndarray): Le vecteur latent de l'individu à muter.

        Retour:
            individual (numpy.ndarray): L'individu muté.
        """
        for i in range(self.latent_dim):
            if np.random.rand() < self.mutation_rate:
                # Ajoute une petite quantité de bruit
                individual[i] += np.random.randn() * 0.01
        return individual

    def evolve(self):
        """
        Fait évoluer la population sur une génération, en utilisant l'élitisme et
        la sélection par tournoi pour générer la nouvelle génération d'individus.

        Retour:
            tuple: Le meilleur individu (numpy.ndarray) et son score de fitness (float).
        """
        # 1. Calcule le score de fitness pour toute la population
        fitness_scores = [self.fitness_func(ind) for ind in self.population]

        # 2. Sélectionne les meilleurs individus pour devenir parents (Élitisme + Sélection par tournoi)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Conserve les 2 meilleurs individus (élitisme)
        next_population = [self.population[i] for i in sorted_indices[:2]]

        # 3. Crée le reste de la nouvelle population par croisement et mutation
        while len(next_population) < self.population_size:
            # Sélection par tournoi : choisit 4 individus au hasard et prend les 2 meilleurs comme parents
            tournament = np.random.choice(self.population_size, 4, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            parent_indices = np.argsort(tournament_fitness)[::-1][:2]
            parent1 = self.population[tournament[parent_indices[0]]]
            parent2 = self.population[tournament[parent_indices[1]]]

            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy() # Conserve un parent s'il n'y a pas de croisement

            mutated_child = self._mutate(child)
            next_population.append(mutated_child)
        
        self.population = next_population
        
        # Retourne le meilleur individu et son score pour cette génération
        best_index = sorted_indices[0]
        return self.population[best_index], fitness_scores[best_index]
