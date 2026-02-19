"""Feature engineering modules."""

from src.features.base_features import (
    get_base_features,
    get_target,
    split_features_by_type,
    get_numerical_features,
    get_binary_features,
    get_target_variable,
    get_excluded_variables
)
from src.features.aggregations import (
    create_all_aggregations,
    create_aggregation
)
from src.features.categoricals import (
    create_all_categoricals,
    create_categorical_feature
)
from src.features.association_rules import (
    AssociationRuleFeatureGenerator
)
from src.features.pruning import (
    RareBinaryFeaturePruner
)

__all__ = [
    # Base features
    'get_base_features',
    'get_target',
    'split_features_by_type',
    'get_numerical_features',
    'get_binary_features', 
    'get_target_variable',
    'get_excluded_variables',
    # Aggregations
    'create_all_aggregations',
    'create_aggregation',
    # Categoricals
    'create_all_categoricals',
    'create_categorical_feature',
    # Association rules
    'AssociationRuleFeatureGenerator',
    # Pruning
    'RareBinaryFeaturePruner'
]
