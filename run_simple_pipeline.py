#!/usr/bin/env python
"""
Script de entrada para el pipeline simplificado de shock hemorrágico.
"""
import sys
from pathlib import Path

# Añadir el directorio raíz al path para que pueda encontrar 'src'
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from simple_pipeline import SimpleShockPipeline


def main():
    """Función principal para ejecutar un ejemplo."""
    # Usar un archivo de ejemplo si no se proporciona uno
    data_path = "data/processed/shock.csv"
    output_dir = "output/simple_pipeline"
    
    # Verificar si existe el archivo de ejemplo
    if not Path(data_path).exists():
        print(f"Archivo de datos no encontrado: {data_path}")
        print("Por favor, proporciona un archivo de datos válido.")
        return
    
    # Crear y ejecutar pipeline
    pipeline = SimpleShockPipeline(data_path=data_path, output_dir=output_dir)
    results = pipeline.run_pipeline()
    
    print("\nResumen de resultados:")
    for model, metrics in results.items():
        print(f"  {model}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")


if __name__ == "__main__":
    main()