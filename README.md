Proyecto de Optimización CVRP con H-ACO

Este repositorio contiene la implementación y el análisis experimental de un Algoritmo Híbrido de Colonia de Hormigas (H-ACO) para resolver un Problema de Ruteo de Vehículos con Capacidad (CVRP) del mundo real.

El proyecto incluye:

Una aplicación interactiva de Streamlit para ejecutar simulaciones y visualizar resultados.

Implementaciones de los algoritmos H-ACO (propuesto), Algoritmo Genético (GA) y Clarke & Wright Savings (CWS).

El conjunto de datos real de 30 paradas y 10 escenarios de simulación.

Cómo Ejecutar

Clonar el repositorio:

git clone [URL-DE-TU-REPO]
cd [NOMBRE-DE-TU-REPO]


Crear un entorno virtual (recomendado):

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate


Instalar las dependencias:

pip install -r requirements.txt


Ejecutar la aplicación Streamlit:

streamlit run streamlit_app.py
