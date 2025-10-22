import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# Importar módulos del proyecto
from src.data_loader import load_customer_data, get_simulation_scenarios, setup_problem_instance
from src.utils import calculate_solution_cost, plot_routes
from src.algorithms.cws import run_cws
from src.algorithms.ga import GeneticAlgorithm
from src.algorithms.h_aco import HybridACO

# Configuración de la página
st.set_page_config(layout="wide", page_title="Optimización CVRP (H-ACO)")

st.title("Panel de Control: Optimización de Rutas (CVRP)")
st.write("Análisis comparativo de H-ACO (Propuesto) vs. GA y CWS para el caso de estudio.")

# --- Cargar Datos Globales ---
try:
    all_customers_df = load_customer_data()
    scenarios = get_simulation_scenarios()
    scenario_names = list(scenarios.keys())
except Exception as e:
    st.error(f"Error cargando los datos: {e}")
    st.stop()

# ==============================================================================
# BARRA LATERAL (CONTROLES)
# ==============================================================================
st.sidebar.header("Configuración de Simulación")

# --- Sección 1: Simulación Única (Visual) ---
st.sidebar.markdown("### 1. Ejecución Visual Única")
selected_scenario_name = st.sidebar.selectbox(
    "Seleccionar Simulación (Instancia)",
    scenario_names,
    index=2 # Default a S-3 por ser la más compleja
)

col_cws, col_ga, col_haco = st.sidebar.columns(3)
run_cws_flag = col_cws.checkbox("CWS", value=True)
run_ga_flag = col_ga.checkbox("GA", value=True)
run_haco_flag = col_haco.checkbox("H-ACO", value=True)

start_single_run = st.sidebar.button("INICIAR EJECUCIÓN VISUAL", type="primary")

st.sidebar.divider()

# --- Sección 2: Parámetros H-ACO ---
st.sidebar.markdown("### 2. Parámetros H-ACO")
n_iterations = st.sidebar.number_input("Iteraciones (n_iterations)", min_value=1, value=100)
n_ants = st.sidebar.number_input("Hormigas (n_ants)", min_value=1, value=20)
alpha = st.sidebar.slider("Influencia Feromona (α)", 0.1, 5.0, 1.0, 0.1)
beta = st.sidebar.slider("Influencia Heurística (β)", 0.1, 10.0, 5.0, 0.1)
rho = st.sidebar.slider("Tasa Evaporación (ρ)", 0.01, 0.5, 0.1, 0.01)

# Parámetros GA (simplificado)
ga_generations = n_iterations # Usar el mismo número de iteraciones
ga_pop_size = n_ants * 2      # Usar una población comparable

st.sidebar.divider()

# --- Sección 3: Experimento Robusto (Paso 4) ---
st.sidebar.markdown("### 3. Experimento Robusto (Paso 4)")
n_runs = st.sidebar.number_input("Número de Corridas (N)", min_value=1, value=10) # Default 10 para rapidez, cambiar a 30
run_statistical_experiment = st.sidebar.button("INICIAR EXPERIMENTO ESTADÍSTICO")

# ==============================================================================
# LÓGICA PRINCIPAL
# ==============================================================================

# Cargar la instancia del problema basado en la selección
try:
    customer_ids_to_visit = scenarios[selected_scenario_name]
    problem_instance = setup_problem_instance(all_customers_df, customer_ids_to_visit)
    st.subheader(f"Instancia: {selected_scenario_name} ({problem_instance['num_nodes']-1} paradas)")
except Exception as e:
    st.error(f"Error preparando la instancia del problema: {e}")
    st.stop()

# --- LÓGICA PARA EJECUCIÓN VISUAL ÚNICA ---
if start_single_run:
    st.header("Resultados de la Ejecución Visual Única")
    
    # Preparar columnas para resultados
    alg_columns = st.columns([1, 1, 1])
    col_map = {
        'CWS': alg_columns[0],
        'GA': alg_columns[1],
        'H-ACO': alg_columns[2]
    }

    # --- Ejecutar CWS ---
    if run_cws_flag:
        with col_map['CWS']:
            st.markdown("#### 1. Clarke & Wright (CWS)")
            with st.spinner("Ejecutando CWS..."):
                start_time = time.time()
                cws_solution, cws_cost = run_cws(problem_instance)
                exec_time = time.time() - start_time
            
            st.metric("Costo Total (Distancia Km)", f"{cws_cost:,.2f} Km")
            st.caption(f"Tiempo: {exec_time:.2f} seg. | Rutas: {len(cws_solution)}")
            
            fig = plot_routes(cws_solution, problem_instance, "Rutas CWS")
            st.plotly_chart(fig, use_container_width=True)

    # --- Ejecutar GA ---
    if run_ga_flag:
        with col_map['GA']:
            st.markdown("#### 2. Algoritmo Genético (GA)")
            with st.spinner(f"Ejecutando GA ({ga_generations} gen)..."):
                start_time = time.time()
                ga = GeneticAlgorithm(
                    problem=problem_instance,
                    pop_size=ga_pop_size,
                    generations=ga_generations,
                    cx_rate=0.8,
                    mut_rate=0.1
                )
                ga_solution, ga_cost = ga.run()
                exec_time = time.time() - start_time
            
            st.metric("Costo Total (Distancia Km)", f"{ga_cost:,.2f} Km")
            st.caption(f"Tiempo: {exec_time:.2f} seg. | Rutas: {len(ga_solution)}")
            
            fig = plot_routes(ga_solution, problem_instance, "Rutas GA")
            st.plotly_chart(fig, use_container_width=True)

    # --- Ejecutar H-ACO ---
    if run_haco_flag:
        with col_map['H-ACO']:
            st.markdown("#### 3. H-ACO (Propuesto)")
            with st.spinner(f"Ejecutando H-ACO ({n_iterations} iter)..."):
                start_time = time.time()
                h_aco = HybridACO(
                    problem=problem_instance,
                    n_ants=n_ants,
                    n_iterations=n_iterations,
                    alpha=alpha,
                    beta=beta,
                    rho=rho,
                    q=100 # Constante Q, se puede sintonizar
                )
                haco_solution, haco_cost = h_aco.run()
                exec_time = time.time() - start_time
            
            st.metric("Costo Total (Distancia Km)", f"{haco_cost:,.2f} Km")
            st.caption(f"Tiempo: {exec_time:.2f} seg. | Rutas: {len(haco_solution)}")
            
            fig = plot_routes(haco_solution, problem_instance, "Rutas H-ACO")
            st.plotly_chart(fig, use_container_width=True)


# --- LÓGICA PARA EXPERIMENTO ESTADÍSTICO ---
if run_statistical_experiment:
    st.header(f"Resultados del Experimento Robusto ({n_runs} corridas)")
    st.write(f"Comparando H-ACO, GA y CWS para la instancia: **{selected_scenario_name}**")
    
    results_list = []
    progress_bar = st.progress(0, text="Iniciando experimento...")
    
    # --- 1. CWS (Solo 1 corrida, es determinista) ---
    cws_solution, cws_cost = run_cws(problem_instance)
    for i in range(n_runs):
        results_list.append({'Algorithm': 'CWS', 'Run': i+1, 'Cost': cws_cost})

    # --- 2. GA y H-ACO (N corridas) ---
    for i in range(n_runs):
        text = f"Ejecutando corrida {i+1}/{n_runs}..."
        progress_bar.progress((i+1)/n_runs, text=text)
        
        # Ejecutar GA
        ga = GeneticAlgorithm(problem_instance, ga_pop_size, ga_generations)
        _, ga_cost = ga.run()
        results_list.append({'Algorithm': 'GA', 'Run': i+1, 'Cost': ga_cost})
        
        # Ejecutar H-ACO
        h_aco = HybridACO(problem_instance, n_ants, n_iterations, alpha, beta, rho)
        _, haco_cost = h_aco.run()
        results_list.append({'Algorithm': 'H-ACO', 'Run': i+1, 'Cost': haco_cost})

    progress_bar.empty()
    st.success("Experimento completado.")
    
    # Crear DataFrame
    df_results = pd.DataFrame(results_list)
    
    # --- Mostrar Resultados ---
    col_stats, col_plot = st.columns(2)
    
    with col_stats:
        st.subheader("Estadísticas Descriptivas")
        df_summary = df_results.groupby('Algorithm')['Cost'].agg(
            ['mean', 'std', 'min', 'max']
        ).reset_index()
        df_summary = df_summary.sort_values(by='mean')
        st.dataframe(df_summary.style.format({
            'mean': '{:,.2f}',
            'std': '{:,.2f}',
            'min': '{:,.2f}',
            'max': '{:,.2f}'
        }))
        
        # --- Análisis Estadístico (Wilcoxon) ---
        st.subheader("Análisis Estadístico (p-values)")
        st.markdown(f"Comparando contra H-ACO (N={n_runs})")
        
        try:
            h_aco_runs = df_results[df_results['Algorithm'] == 'H-ACO']['Cost']
            ga_runs = df_results[df_results['Algorithm'] == 'GA']['Cost']
            cws_runs = df_results[df_results['Algorithm'] == 'CWS']['Cost']
            
            # H-ACO vs GA
            stat_ga, p_ga = stats.wilcoxon(h_aco_runs, ga_runs, alternative='less')
            st.metric(
                label="p-value (H-ACO vs. GA)", 
                value=f"{p_ga:.4e}",
                help="Prueba si H-ACO es significativamente *menor* que GA."
            )
            
            # H-ACO vs CWS
            stat_cws, p_cws = stats.wilcoxon(h_aco_runs, cws_runs, alternative='less')
            st.metric(
                label="p-value (H-ACO vs. CWS)", 
                value=f"{p_cws:.4e}",
                help="Prueba si H-ACO es significativamente *menor* que CWS."
            )
            
            st.caption("Un p-value < 0.05 indica una diferencia estadísticamente significativa.")
            
        except Exception as e:
            st.error(f"Error en el test estadístico: {e}")
            st.caption("Asegúrate de tener N > 1 y varianza en los resultados.")

    with col_plot:
        st.subheader("Distribución de Costos (Box Plot)")
        fig = px.box(
            df_results, 
            x='Algorithm', 
            y='Cost', 
            color='Algorithm',
            title=f"Comparación de Costos en {n_runs} corridas ({selected_scenario_name})",
            points="all"
        )
        fig.update_layout(xaxis_title="Algoritmo", yaxis_title="Costo Total (Distancia Km)")
        st.plotly_chart(fig, use_container_width=True)
