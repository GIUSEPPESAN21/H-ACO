import pandas as pd
import numpy as np
from src.utils import get_haversine_distance

# Constantes extraídas del documento
VEHICLE_CAPACITY = 150
DEPOT_COORDS = (3.88010, -76.29842) # Guadalajara de Buga (Confirmado por usuario)

def load_customer_data():
    """Carga los 30 clientes de la Tabla 1 del documento."""
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'name': [
            "CD Bogotá – Fontibón (Free Trade Zone)", "CD Bogotá – Siberia (Industrial Park)",
            "CD Medellín – Guayabal (Industrial Zone)", "CD Medellín – Itagüí (Industrial Center)",
            "CD Cali – Yumbo (Arroyohondo)", "CD Cali – Acopi (Industrial Zone)",
            "CD Barranquilla – Vía 40 (Industrial Zone)", "CD Barranquilla – Galapa (Industrial Park)",
            "CD Cartagena – Mamonal (Industrial Zone)", "CD Cartagena – Bosque (Industrial Zone)",
            "CD Buenaventura – Port Authority", "CD Santa Marta – Tayrona (Free Trade Zone)",
            "CD Bucaramanga – Girón (Industrial Park)", "CD Cúcuta – (Free Trade Zone)",
            "CD Pereira – Dosquebradas (La Romelia)", "CD Manizales – Juanchito (Industrial Zone)",
            "CD Ibagué – El Papayo (Industrial Zone)", "CD Neiva – Surabastos / Industrial Zone",
            "CD Villavicencio – Ocoa (Industrial Park)", "CD Pasto – Catambuco (Industrial Zone)",
            "CD Tunja – (Industrial Park)", "CD Armenia – Coffee Axis (Free Trade Zone) (La Tebaida)",
            "CD Montería – (Industrial Zone)", "CD Valledupar – (Industrial Zone)",
            "CD Riohacha – (Free Trade Zone)", "CD Sincelejo – (Industrial Park)",
            "CD Tocancipá (Cund) – (Industrial Park)", "CD Mosquera (Cund) – (Industrial Park)",
            "CD Malambo (Atlántico) – Pimsa (Industrial Park)", "CD Palmira (Valle) – Palmaseca (Free Trade Zone)"
        ],
        'lat': [
            4.685, 4.750, 6.210, 6.170, 3.530, 3.500, 10.990, 10.880, 10.350, 10.390,
            3.880, 11.200, 7.060, 7.910, 4.830, 5.040, 4.410, 2.900, 4.100, 1.180,
            5.550, 4.460, 8.730, 10.450, 11.530, 9.280, 4.900, 4.710, 10.850, 3.570
        ],
        'lon': [
            -74.140, -74.180, -75.590, -75.620, -76.450, -76.500, -74.800, -74.870, -75.500, -75.520,
            -77.030, -74.190, -73.150, -72.520, -75.730, -75.470, -75.200, -75.280, -73.580, -77.280,
            -73.340, -75.780, -75.900, -73.260, -72.910, -75.400, -74.000, -74.210, -74.780, -76.280
        ],
        'demand': [
            65, 72, 58, 45, 60, 50, 42, 33, 75, 38, 78, 28, 35, 23, 30, 18, 15, 12, 17, 11,
            20, 48, 26, 14, 13, 11, 70, 68, 62, 55
        ]
    }
    customers_df = pd.DataFrame(data).set_index('id')
    return customers_df

def get_simulation_scenarios():
    """Carga las 10 simulaciones de la Tabla 2 del documento."""
    scenarios = {
        'S-1': [1, 3, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 4],
        'S-2': [2, 3, 5, 6, 7, 9, 11, 13, 14, 15, 22, 23, 27, 28, 29, 30, 1, 10],
        'S-3': list(range(1, 31)), # Todos los 30
        'S-4': [1, 2, 3, 4, 5, 6, 9, 11, 22, 27, 28, 29, 30, 7, 8],
        'S-5': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 22, 23, 24, 27, 28, 29, 30, 4, 13, 14],
        'S-6': [i for i in range(1, 31) if i not in [20, 26]],
        'S-7': [1, 2, 3, 4, 5, 7, 9, 11, 13, 21, 22, 27, 28, 29, 30, 6, 8],
        'S-8': [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 22, 27, 28, 29, 30, 8, 12, 14],
        'S-9': [i for i in range(1, 31) if i not in [16, 18, 20, 25]],
        'S-10': [1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 22, 23, 27, 28, 29, 30, 4, 8, 10]
    }
    return scenarios

def setup_problem_instance(all_customers_df, customer_ids_to_visit):
    """
    Prepara la instancia del problema para una simulación específica.
    Crea la matriz de distancias y el vector de demandas.
    """
    
    # Filtrar clientes para esta simulación
    instance_customers = all_customers_df.loc[customer_ids_to_visit].copy()
    
    # Mapeo de ID de cliente (1-30) a índice de matriz (1-N)
    # El Depósito (ID 0) es SIEMPRE el índice 0
    id_to_matrix_idx = {cid: i+1 for i, cid in enumerate(instance_customers.index)}
    matrix_idx_to_id = {i+1: cid for i, cid in enumerate(instance_customers.index)}
    
    # Coordenadas (el Depósito es el índice 0)
    coords = {0: DEPOT_COORDS}
    for cid, i in id_to_matrix_idx.items():
        coords[i] = (instance_customers.loc[cid, 'lat'], instance_customers.loc[cid, 'lon'])
    
    num_nodes = len(instance_customers) + 1 # +1 por el depósito
    
    # Crear vector de demandas (el depósito tiene demanda 0)
    demands = np.zeros(num_nodes)
    for cid, i in id_to_matrix_idx.items():
        demands[i] = instance_customers.loc[cid, 'demand']

    # Crear matriz de distancias (costo)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = get_haversine_distance(
                coords[i][0], coords[i][1],
                coords[j][0], coords[j][1]
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    problem = {
        'num_nodes': num_nodes,
        'demands': demands,
        'dist_matrix': dist_matrix,
        'capacity': VEHICLE_CAPACITY,
        'coords': coords, # Coordenadas (Lat, Lon) por índice de matriz
        'id_to_idx': id_to_matrix_idx,
        'idx_to_id': matrix_idx_to_id,
        'customer_nodes': list(range(1, num_nodes)) # Índices de clientes (excl. depósito)
    }
    return problem
