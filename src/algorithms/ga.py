import random
import numpy as np
from src.utils import calculate_solution_cost

class GeneticAlgorithm:
    """Implementación de un GA estándar (Benchmark 2)."""
    
    def __init__(self, problem, pop_size=100, generations=200, cx_rate=0.8, mut_rate=0.1):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        
        self.dist_matrix = problem['dist_matrix']
        self.demands = problem['demands']
        self.capacity = problem['capacity']
        self.customer_nodes = problem['customer_nodes'].copy()

    def _create_individual(self):
        """Crea un cromosoma (una permutación aleatoria de clientes)."""
        individual = self.customer_nodes.copy()
        random.shuffle(individual)
        return individual

    def _decode_chromosome(self, chromosome):
        """Divide un cromosoma (lista) en rutas (lista de listas) basado en capacidad."""
        solution = []
        current_route = []
        current_load = 0
        
        for node in chromosome:
            demand = self.demands[node]
            if current_load + demand <= self.capacity:
                current_route.append(node)
                current_load += demand
            else:
                if current_route: # Evitar rutas vacías si una parada ya supera la capacidad
                    solution.append(current_route)
                current_route = [node]
                current_load = demand
        
        if current_route: # Añadir la última ruta
            solution.append(current_route)
            
        return solution

    def _calculate_fitness(self, chromosome):
        """Calcula el fitness (costo total) de un cromosoma."""
        solution = self._decode_chromosome(chromosome)
        cost = calculate_solution_cost(solution, self.dist_matrix)
        return cost, solution

    def _selection(self, population):
        """Selección por torneo."""
        tournament_size = 3
        selected = []
        for _ in range(self.pop_size):
            aspirants = random.sample(population, tournament_size)
            aspirants.sort(key=lambda x: x['fitness']) # Minimización
            selected.append(aspirants[0]['chromosome'])
        return selected

    def _crossover(self, parent1, parent2):
        """Order Crossover (OX1)."""
        size = len(parent1)
        child1, child2 = [None]*size, [None]*size
        
        start, end = sorted(random.sample(range(size), 2))
        
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        p1_idx, p2_idx = (end + 1) % size, (end + 1) % size
        c1_idx, c2_idx = (end + 1) % size, (end + 1) % size
        
        p1_items_in_child2 = set(child2)
        p2_items_in_child1 = set(child1)
        
        for _ in range(size - (end - start + 1)):
            while parent2[p2_idx] in p2_items_in_child1:
                p2_idx = (p2_idx + 1) % size
            child1[c1_idx] = parent2[p2_idx]
            c1_idx = (c1_idx + 1) % size

            while parent1[p1_idx] in p1_items_in_child2:
                p1_idx = (p1_idx + 1) % size
            child2[c2_idx] = parent1[p1_idx]
            c2_idx = (c2_idx + 1) % size
            
        return child1, child2

    def _mutation(self, chromosome):
        """Mutación por intercambio (Swap)."""
        if random.random() < self.mut_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def run(self):
        # 1. Inicializar población
        population = []
        for _ in range(self.pop_size):
            chromo = self._create_individual()
            fitness, solution = self._calculate_fitness(chromo)
            population.append({'chromosome': chromo, 'fitness': fitness, 'solution': solution})
            
        best_ever = min(population, key=lambda x: x['fitness'])
        best_solution = best_ever['solution']
        best_cost = best_ever['fitness']

        # 2. Evolucionar por N generaciones
        for _ in range(self.generations):
            # 1. Selección
            selected_parents = self._selection(population)
            
            # 2. Cruce y Mutación
            new_population_chromos = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected_parents[i], selected_parents[i+1]
                c1, c2 = (self._crossover(p1, p2)) if random.random() < self.cx_rate else (p1[:], p2[:])
                new_population_chromos.extend([self._mutation(c1), self._mutation(c2)])
                
            # 3. Evaluar nueva población y reemplazar
            new_population = []
            for chromo in new_population_chromos:
                fitness, solution = self._calculate_fitness(chromo)
                new_population.append({'chromosome': chromo, 'fitness': fitness, 'solution': solution})
            
            # Reemplazo (Elitismo: mantener la mejor solución)
            new_population.sort(key=lambda x: x['fitness'])
            
            if new_population[0]['fitness'] < best_cost:
                best_cost = new_population[0]['fitness']
                best_solution = new_population[0]['solution']
            
            # Reemplazar la peor de la nueva gen con la mejor de la anterior
            new_population[-1] = {'chromosome': best_ever['chromosome'], 'fitness': best_cost, 'solution': best_solution}
            population = new_population
        
        return best_solution, best_cost
