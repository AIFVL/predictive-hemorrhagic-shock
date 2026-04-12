"""Data loading utilities for various file formats."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.utils import logger


@dataclass(frozen=True, kw_only=True)
class FileFormatSpec:
    display_name: str
    read_function: Callable[..., Any]
    write_function: Callable[..., Any]
    read_options: dict[str, Any]
    write_options: dict[str, Any]


class DataLoader:
    """Utility class for loading and saving files by extension."""

    _FORMATS = {
        '.csv': FileFormatSpec(
            display_name='CSV',
            read_function=pd.read_csv,
            write_function=pd.DataFrame.to_csv,
            read_options={'sep': ','},
            write_options={'index': False, 'header': True},
        ),
        '.parquet': FileFormatSpec(
            display_name='Parquet',
            read_function=pd.read_parquet,
            write_function=pd.DataFrame.to_parquet,
            read_options={},
            write_options={},
        ),
        '.json': FileFormatSpec(
            display_name='JSON',
            read_function=lambda path, **options: json.loads(Path(path).read_text()),
            write_function=lambda data, path, **options: Path(path).write_text(
                json.dumps(data, indent=options.pop('indent', 2), **options)
            ),
            read_options={},
            write_options={'indent': 2},
        ),
        '.xlsx': FileFormatSpec(
            display_name='Excel',
            read_function=pd.read_excel,
            write_function=pd.DataFrame.to_excel,
            read_options={'sheet_name': 0},
            write_options={'index': False},
        ),
        '.xls': FileFormatSpec(
            display_name='Excel',
            read_function=pd.read_excel,
            write_function=pd.DataFrame.to_excel,
            read_options={'sheet_name': 0},
            write_options={'index': False},
        ),
    }

    @classmethod
    def get_format_spec(cls, path: str | Path) -> FileFormatSpec:
        file_path = Path(path)
        format_spec = cls._FORMATS.get(file_path.suffix.lower())
        if format_spec is None:
            raise ValueError(f"Unsupported file format: {file_path.suffix.lower()}")
        return format_spec

    @classmethod
    def load(cls, path: str | Path, **options) -> Any:
        file_path = Path(path)
        format_spec = cls.get_format_spec(file_path)

        load_options = {**format_spec.read_options, **options}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info({
            'event': 'data_load_started',
            'path': str(file_path),
            'format': format_spec.display_name,
        })
        data = format_spec.read_function(file_path, **load_options)
        logger.info({
            'event': 'data_load_completed',
            'path': str(file_path),
            'format': format_spec.display_name,
        })
        return data

    @classmethod
    def save(cls, data: Any, path: str | Path, **options) -> None:
        file_path = Path(path)
        format_spec = cls.get_format_spec(file_path)

        save_options = {**format_spec.write_options, **options}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info({
            'event': 'data_save_started',
            'path': str(file_path),
            'format': format_spec.display_name,
            'data_type': type(data).__name__,
        })

        format_spec.write_function(data, file_path, **save_options)

        logger.info({
            'event': 'data_save_completed',
            'path': str(file_path),
            'format': format_spec.display_name,
            'data_type': type(data).__name__,
        })