import time
from src.utils import calculate_solution_cost

def run_cws(problem):
    """
    Implementación de la heurística CWS (Benchmark 1).
    """
    start_time = time.time()
    
    dist_matrix = problem['dist_matrix']
    demands = problem['demands']
    capacity = problem['capacity']
    customer_nodes = problem['customer_nodes'] # Índices 1 a N
    
    # 1. Calcular ahorros (savings)
    savings = []
    for i in customer_nodes:
        for j in customer_nodes:
            if i < j:
                s_ij = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
                if s_ij > 0:
                    savings.append((s_ij, i, j))
                
    # 2. Ordenar ahorros de mayor a menor
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Inicializar rutas (una por cliente)
    routes = {i: [i] for i in customer_nodes}
    route_demands = {i: demands[i] for i in customer_nodes}
    
    # 4. Fusionar rutas
    for s_ij, i, j in savings:
        
        # Encontrar a qué rutas pertenecen i y j
        route_i_key, route_j_key = None, None
        is_i_edge, is_j_edge = False, False # True si está en un extremo de la ruta
        
        for key, route in routes.items():
            if not route: continue
            if route[0] == i:
                route_i_key = key
                is_i_edge = True
            elif route[-1] == i:
                route_i_key = key
                is_i_edge = True
            
            if route[0] == j:
                route_j_key = key
                is_j_edge = True
            elif route[-1] == j:
                route_j_key = key
                is_j_edge = True

        # Solo fusionar si:
        # 1. i y j están en rutas DIFERENTES
        # 2. i y j están en los extremos de sus rutas
        if route_i_key is not None and route_j_key is not None and \
           route_i_key != route_j_key and is_i_edge and is_j_edge:
            
            route_i = routes[route_i_key]
            route_j = routes[route_j_key]
            
            # Verificar capacidad
            if route_demands[route_i_key] + route_demands[route_j_key] > capacity:
                continue
            
            new_route = None
            if route_i[-1] == i and route_j[0] == j:
                new_route = route_i + route_j
            elif route_j[-1] == j and route_i[0] == i:
                new_route = route_j + route_i
            elif route_i[0] == i and route_j[0] == j:
                new_route = list(reversed(route_i)) + route_j
            elif route_i[-1] == i and route_j[-1] == j:
                 new_route = route_i + list(reversed(route_j))
            
            if new_route:
                new_demand = route_demands[route_i_key] + route_demands[route_j_key]
                routes[route_i_key] = new_route
                route_demands[route_i_key] = new_demand
                
                routes[route_j_key] = [] # Marcar ruta j como vacía
                route_demands[route_j_key] = 0

    # 5. Formatear solución final
    final_solution = [route for route in routes.values() if route]
    best_cost = calculate_solution_cost(final_solution, dist_matrix)
    
    return final_solution, best_cost
