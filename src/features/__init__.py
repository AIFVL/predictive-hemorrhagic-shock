"""Feature engineering modules."""

from src.features.base_features import (
    get_base_features,
    get_target,
    split_features_by_type,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE
)
from src.features.aggregations import (
    create_all_aggregations,
    create_carga_comorbilidades,
    create_riesgo_cardiovascular,
    create_riesgo_respiratorio,
    create_riesgo_metabolico
)
from src.features.categoricals import (
    create_all_categoricals,
    create_categoria_edad,
    create_categoria_hb_preqx
)

__all__ = [
    # Base features
    'get_base_features',
    'get_target',
    'split_features_by_type',
    'NUMERICAL_FEATURES',
    'BINARY_FEATURES', 
    'CATEGORICAL_FEATURES',
    'TARGET_VARIABLE',
    # Aggregations
    'create_all_aggregations',
    'create_carga_comorbilidades',
    'create_riesgo_cardiovascular',
    'create_riesgo_respiratorio',
    'create_riesgo_metabolico',
    # Categoricals
    'create_all_categoricals',
    'create_categoria_edad',
    'create_categoria_hb_preqx'
]
