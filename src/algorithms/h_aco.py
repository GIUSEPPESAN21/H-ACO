import random
import numpy as np
from src.utils import calculate_route_cost, calculate_solution_cost

class HybridACO:
    """Implementación de H-ACO (Algoritmo Propuesto)."""
    
    def __init__(self, problem, n_ants, n_iterations, alpha, beta, rho, q=100):
        self.problem = problem
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha   # Influencia feromona
        self.beta = beta     # Influencia heurística (distancia)
        self.rho = rho       # Tasa de evaporación
        self.q = q           # Constante de depósito de feromona
        
        self.dist_matrix = problem['dist_matrix']
        self.demands = problem['demands']
        self.capacity = problem['capacity']
        self.customer_nodes = problem['customer_nodes']
        self.n_nodes = problem['num_nodes']
        
        # Inicializar feromonas
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))
        
        # Inicializar heurística (inverso de la distancia)
        self.heuristic = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and self.dist_matrix[i, j] > 0:
                    self.heuristic[i, j] = 1.0 / self.dist_matrix[i, j]

        self.best_solution = None
        self.best_cost = float('inf')

    def run(self):
        for _ in range(self.n_iterations):
            all_ant_solutions = []
            
            for _ in range(self.n_ants):
                # 1. Construir solución
                ant_solution = self._construct_solution()
                
                # 2. Hibridación: Aplicar VNS (Búsqueda Local)
                ant_solution = self._apply_vns(ant_solution)
                
                ant_cost = calculate_solution_cost(ant_solution, self.dist_matrix)
                
                all_ant_solutions.append((ant_solution, ant_cost))
                
                if ant_cost < self.best_cost:
                    self.best_solution = ant_solution
                    self.best_cost = ant_cost
            
            # 3. Actualizar Feromonas
            self._update_pheromones(all_ant_solutions)
            
        return self.best_solution, self.best_cost

    def _construct_solution(self):
        """Una hormiga construye una solución completa (múltiples rutas)."""
        solution = []
        unvisited = self.customer_nodes.copy()
        
        while unvisited:
            current_route = []
            current_load = 0
            current_node = 0 # Empezar en el depósito
            
            while True:
                # Encontrar siguientes paradas factibles
                feasible_next = []
                for node in unvisited:
                    if current_load + self.demands[node] <= self.capacity:
                        feasible_next.append(node)
                
                if not feasible_next:
                    break # Ruta llena, volver al depósito
                    
                # Calcular probabilidades
                probs = self._calculate_probabilities(current_node, feasible_next)
                
                # Seleccionar siguiente nodo (ruleta)
                next_node = random.choices(feasible_next, weights=probs, k=1)[0]
                
                current_route.append(next_node)
                current_load += self.demands[next_node]
                unvisited.remove(next_node)
                current_node = next_node
            
            if current_route:
                solution.append(current_route)
                
        return solution

    def _calculate_probabilities(self, current_node, feasible_nodes):
        probs = []
        for node in feasible_nodes:
            tau = self.pheromone[current_node, node] ** self.alpha
            eta = self.heuristic[current_node, node] ** self.beta
            probs.append(tau * eta)
            
        total_prob = sum(probs)
        if total_prob == 0: # Si no hay feromona/heurística, elegir al azar
            return [1.0 / len(feasible_nodes)] * len(feasible_nodes)
            
        return [p / total_prob for p in probs]

    def _apply_vns(self, solution):
        """
        Aplica Variable Neighborhood Search (VNS) para mejorar la solución de la hormiga.
        Esta es la parte "Híbrida" (H-ACO).
        """
        improved_solution = solution
        cost_solution = calculate_solution_cost(improved_solution, self.dist_matrix)

        # Definir vecindarios (simplificado: 2-opt intra-ruta y re-inserción inter-ruta)
        neighborhoods = [self._vns_2opt, self._vns_relocate]
        k = 0
        while k < len(neighborhoods):
            new_solution, new_cost = neighborhoods[k](improved_solution, cost_solution)
            if new_cost < cost_solution:
                improved_solution = new_solution
                cost_solution = new_cost
                k = 0 # Volver al primer vecindario
            else:
                k += 1 # Probar siguiente vecindario
                
        return improved_solution

    def _vns_2opt(self, solution, original_cost):
        """Neighborhood 1: 2-opt (Intra-Ruta)."""
        best_solution = [r[:] for r in solution] # Copia profunda
        best_cost = original_cost
        
        for r_idx, route in enumerate(best_solution):
            if len(route) < 2: continue
            
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # [..., i, i+1, ..., j, j+1, ...] -> [..., i, j, ..., i+1, j+1, ...]
                    new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
                    
                    # Calcular nuevo costo (delta)
                    old_cost = calculate_route_cost(route, self.dist_matrix)
                    new_cost = calculate_route_cost(new_route, self.dist_matrix)
                    
                    if new_cost < old_cost:
                        delta_cost = new_cost - old_cost
                        best_solution[r_idx] = new_route
                        best_cost += delta_cost
                        return best_solution, best_cost # Retornar en la primera mejora
        return best_solution, best_cost # Retornar la original si no hay mejora

    def _vns_relocate(self, solution, original_cost):
        """Neighborhood 2: Re-inserción (Inter-Ruta)."""
        best_solution = [r[:] for r in solution]
        best_cost = original_cost
        
        for r1_idx, route1 in enumerate(best_solution):
            for node_idx, node_to_move in enumerate(route1):
                
                # Probar mover 'node_to_move' a otra ruta
                for r2_idx, route2 in enumerate(best_solution):
                    if r1_idx == r2_idx: continue
                    
                    # Verificar capacidad
                    route2_load = sum(self.demands[n] for n in route2)
                    if route2_load + self.demands[node_to_move] > self.capacity:
                        continue
                        
                    # Probar insertar en cada posición de route2
                    for insert_pos in range(len(route2) + 1):
                        new_route1 = route1[:node_idx] + route1[node_idx+1:]
                        new_route2 = route2[:insert_pos] + [node_to_move] + route2[insert_pos:]
                        
                        # Calcular costo delta
                        old_cost = calculate_route_cost(route1, self.dist_matrix) + \
                                   calculate_route_cost(route2, self.dist_matrix)
                        new_cost = calculate_route_cost(new_route1, self.dist_matrix) + \
                                   calculate_route_cost(new_route2, self.dist_matrix)
                                   
                        if new_cost < old_cost:
                            new_solution_set = [r[:] for r in solution]
                            new_solution_set[r1_idx] = new_route1
                            new_solution_set[r2_idx] = new_route2
                            # Limpiar rutas vacías
                            new_solution_set = [r for r in new_solution_set if r]
                            
                            delta_cost = new_cost - old_cost
                            return new_solution_set, original_cost + delta_cost # Retornar en primera mejora

        return best_solution, best_cost # Retornar la original si no hay mejora

    def _update_pheromones(self, all_ant_solutions):
        # 1. Evaporación
        self.pheromone *= (1.0 - self.rho)
        
        # 2. Depósito (basado en la calidad de la solución)
        for solution, cost in all_ant_solutions:
            pheromone_deposit = self.q / cost
            
            for route in solution:
                # Depósito -> Primer cliente
                self.pheromone[0, route[0]] += pheromone_deposit
                # Cliente -> Cliente
                for i in range(len(route) - 1):
                    self.pheromone[route[i], route[i+1]] += pheromone_deposit
                # Último cliente -> Depósito
                self.pheromone[route[-1], 0] += pheromone_deposit
