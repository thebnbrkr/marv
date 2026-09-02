from .arch import ArchAdapter, LlamaStyleFFN, detect_adapter
from .extract import VindexLite, default_layer_bands, extract, extract_streaming
from .probe import (
    Association,
    build_down_meta,
    describe,
    describe_entity,
    describe_feature,
    logit_lens,
    top_features,
)
from .context import describe_prompt, hidden_states_at_layers
from .diff import FeatureDelta, diff, most_changed, per_layer_score
from .edit import ablate, constellation, restore, steer, suppress
from .evaluate import (
    BatteryDiff,
    BatteryResult,
    Probe,
    diff_battery,
    run_battery,
    study_edit,
    suppression_frontier,
)
from .heatmap import ActivationMatrix, activation_matrix, polysemantic_features
from .layer_heatmap import LayerFeatureHeatmap, layer_attribution, top_features_per_layer
from .layer_heatmap import compute as layer_feature_heatmap
from .layer_heatmap import difference as layer_heatmap_difference
from .layer_heatmap import layer_trace, peak_activation_trace, plot_comparison
from .clustering import PromptActivations, cluster_features, prompt_activations, reduce_pca, reduce_tsne

__all__ = [
    # architecture + extraction
    "ArchAdapter",
    "LlamaStyleFFN",
    "detect_adapter",
    "VindexLite",
    "extract",
    "extract_streaming",
    "default_layer_bands",
    # probe / describe
    "top_features",
    "logit_lens",
    "build_down_meta",
    "describe_feature",
    "describe",
    "describe_entity",
    "Association",
    "describe_prompt",
    "hidden_states_at_layers",
    # diff
    "FeatureDelta",
    "diff",
    "most_changed",
    "per_layer_score",
    # edit + evaluate
    "suppress",
    "ablate",
    "restore",
    "steer",
    "constellation",
    "Probe",
    "run_battery",
    "diff_battery",
    "study_edit",
    "suppression_frontier",
    "BatteryResult",
    "BatteryDiff",
    # heatmaps + clustering
    "ActivationMatrix",
    "activation_matrix",
    "polysemantic_features",
    "LayerFeatureHeatmap",
    "layer_feature_heatmap",
    "layer_heatmap_difference",
    "layer_trace",
    "peak_activation_trace",
    "layer_attribution",
    "top_features_per_layer",
    "plot_comparison",
    "PromptActivations",
    "prompt_activations",
    "reduce_pca",
    "reduce_tsne",
    "cluster_features",
]
