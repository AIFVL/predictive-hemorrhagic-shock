import argparse
import sys
from pathlib import Path

from pipelines.eda_pipeline import EDA_Pipeline


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Análisis Exploratorio de Datos (EDA) para predicción de shock hemorrágico",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../data/processed/shock.csv",
        help="Ruta al archivo CSV de datos",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../reports/eda",
        help="Directorio de salida para resultados del EDA",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Nombre de la columna objetivo (se detecta automáticamente si no se especifica)",
    )

    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()

    # Validar argumentos
    if not Path(args.data).exists():
        print(f"Error: Archivo de datos no encontrado: {args.data}")
        sys.exit(1)

    # Crear y ejecutar pipeline
    print("\n" + "=" * 70)
    print("PIPELINE DE ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("PREDICCIÓN DE SHOCK HEMORRÁGICO")
    print("=" * 70)
    print("\nConfiguración:")
    print(f"  Datos: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Target: {args.target if args.target else 'Auto-detectar'}")
    print()

    pipeline = EDA_Pipeline(data_path=args.data, output_dir=args.output, target_col=args.target)

    try:
        pipeline.run_full_eda()

        print("\n" + "=" * 70)
        print("PIPELINE DE EDA COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("\nAnalisis exploratorio completado")
        print(f"\nRevise los resultados en: {args.output}/")
        print("\nArchivos generados:")
        print("  eda_summary_report.md: Reporte resumen en Markdown")
        print("  tables/: Tablas CSV con resultados estadisticos")
        print("  figures/: Visualizaciones en formato PNG")
        print()
        print("Proximos pasos:")
        print("  1. Revisar el reporte resumen (eda_summary_report.md)")
        print("  2. Analizar las tablas de asociaciones significativas")
        print("  3. Interpretar las visualizaciones generadas")
        print("  4. Identificar variables clave para el modelado")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR EN EL PIPELINE DE EDA")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
