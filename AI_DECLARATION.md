## Declaración de Uso de Inteligencia Artificial Generativa (IAG)

En concordancia con las políticas institucionales de la Universidad Icesi respecto al uso de Inteligencia Artificial Generativa (clasificación de libre uso y co-creación responsable), los autores declaran el alcance de la asistencia tecnológica en el desarrollo de este repositorio:

### 1. Ámbito de Aplicación de la IAG

Se utilizaron modelos de lenguaje de gran tamaño (LLMs) como herramientas de soporte para las siguientes tareas técnicas:

* **Generación de Código Base y Boilerplate:** Asistencia en la escritura de scripts estructurales para la canalización de datos (*pipelines*), orquestación con Apache Airflow y componentes de la interfaz gráfica en Streamlit.
* **Refactorización y Optimización:** Sugerencias de optimización de sintaxis en Python, vectorización de operaciones de datos con Pandas/NumPy y estructuración de consultas de configuración en formato YAML.
* **Depuración (Debugging):** Soporte en la identificación de excepciones, manejo de errores de dependencias y pruebas unitarias preliminares.

### 2. Supervisión, Razonamiento y Autoría Humana

Los autores enfatizan que la IA actuó estrictamente como un asistente técnico ejecutor. El control conceptual y metodológico fue ejercido en su totalidad por los humanos bajo los siguientes ejes:

* **Diseño Arquitectónico:** La definición del patrón de diseño *Factory*, el desacoplamiento de la lógica mediante el `ConfigurationManager` y la estrategia de partición de datos fueron decisiones de diseño de software exclusivas del equipo de desarrollo.
* **Diseño Experimental:** La configuración de los experimentos (como el Experimento 5), la justificación matemática de los hiperparámetros de penalización (`scale_pos_weight`, `class_weight`) y la decisión metodológica de prescindir de SMOTE fueron producto del análisis crítico de los autores.
* **Interpretación Clínica y Estadística:** El análisis de los valores SHAP, la ponderación del *F2-score* y la contextualización médica del shock hemorrágico fueron razonados y validados analíticamente por el equipo investigador, garantizando que el software responda fielmente a la realidad clínica del estudio.

### 3. Responsabilidad Intelectual y Técnica

Los autores asumen la responsabilidad total por el contenido, la fidelidad, la corrección sintáctica y la integridad bioética de todo el código fuente y los artefactos de software compilados en este repositorio. La IA generativa se limitó a acelerar los flujos de implementación, sin sustituir el rigor intelectual de la investigación.
