#!/usr/bin/env python
"""
Script ejecutable para el pipeline de prediccion de shock hemorragico.

Uso:
    python run_shock_pipeline.py
    python run_shock_pipeline.py --data shock.csv --output shock_output
    python run_shock_pipeline.py --help
"""

import argparse
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipelines.shock_pipeline import ShockPredictionPipeline


def parse_args():
    """Parsea argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de prediccion de shock hemorragico",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/shock.csv",
        help="Ruta al archivo CSV de datos"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/shock_model",
        help="Directorio de salida para resultados"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporcion de datos para test (0.0-1.0)"
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Numero de folds para cross-validation"
    )

    parser.add_argument(
        "--min-spec",
        type=float,
        default=0.6,
        help="Especificidad minima para optimizacion de umbral"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad"
    )

    return parser.parse_args()


def main():
    """Funcion principal."""
    args = parse_args()

    # Validar argumentos
    if not Path(args.data).exists():
        print(f"Error: Archivo de datos no encontrado: {args.data}")
        sys.exit(1)

    if args.test_size <= 0 or args.test_size >= 1:
        print(f"Error: test-size debe estar entre 0 y 1, recibido: {args.test_size}")
        sys.exit(1)

    if args.cv_splits < 2:
        print(f"Error: cv-splits debe ser al menos 2, recibido: {args.cv_splits}")
        sys.exit(1)

    # Crear y ejecutar pipeline
    print("\n" + "="*70)
    print("PIPELINE DE PREDICCION DE SHOCK HEMORRAGICO")
    print("="*70)
    print(f"\nConfiguracion:")
    print(f"  Datos: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Test size: {args.test_size}")
    print(f"  CV splits: {args.cv_splits}")
    print(f"  Min specificity: {args.min_spec}")
    print(f"  Seed: {args.seed}")
    print()

    pipeline = ShockPredictionPipeline(
        data_path=args.data,
        output_dir=args.output,
        seed=args.seed
    )

    try:
        pipeline.run_full_pipeline(
            test_size=args.test_size,
            cv_splits=args.cv_splits,
            min_spec=args.min_spec
        )

        print("\n" + "="*70)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\nRevise los resultados en: {args.output}/")
        print("\nArchivos generados:")
        print("  - artifact_<modelo>.joblib: Modelo entrenado")
        print("  - artifact_metadata.json: Metadata del modelo")
        print("  - model_summary.json: Resumen de metricas")
        print("  - *.png: Visualizaciones")
        print("  - explain/: Importancia de features")
        print()

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR EN EL PIPELINE")
        print("="*70)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
