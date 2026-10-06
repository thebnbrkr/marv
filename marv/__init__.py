from ._version import __version__
from .arch import ArchAdapter, LlamaStyleFFN, detect_adapter, register_adapter
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
from .context import (
    ActiveFeature,
    active_features,
    describe_prompt,
    feature_activations_at_layers,
    hidden_states_at_layers,
)
from .diff import FeatureDelta, diff, lineage_score, most_changed, per_layer_score
from .edit import ablate, constellation, restore, steer, suppress
from .evaluate import (
    BatteryDiff,
    BatteryResult,
    Probe,
    diff_battery,
    frontier_table,
    run_battery,
    rank_by_ablation_effect,
    split_probes,
    study_edit,
    suppression_by_layer,
    suppression_frontier,
)
from .heatmap import ActivationMatrix, activation_matrix, polysemantic_features
from .layer_heatmap import LayerFeatureHeatmap, layer_attribution, top_features_per_layer
from .layer_heatmap import compute as layer_feature_heatmap
from .layer_heatmap import difference as layer_heatmap_difference
from .layer_heatmap import layer_trace, peak_activation_trace, plot_comparison
from .clustering import PromptActivations, cluster_features, prompt_activations, reduce_pca, reduce_tsne
from .trace import (
    Decomposition,
    PatchSweep,
    Writes,
    all_components,
    capture_writes,
    component,
    decompose_logit,
    logit_diff_metric,
    mean_ablate,
    mean_writes,
    patch_sweep,
    replace_outputs,
    trace_by_depth,
)
from .diagnostics import (
    Health,
    ScaleCheck,
    compare_scale,
    dead_features,
    find_bottlenecks,
    health,
    load_bearing,
    null_model,
    write_norms,
)
from .history import (
    BatteryTest,
    History,
    Test,
    changed_neurons,
    checkpoint_loader,
    history_callback,
    revert_neurons,
)
from .batteries import (
    WORLD_CAPITALS,
    broad_controls,
    capital_edit_battery,
    capital_probes,
    domain_probes,
)

__all__ = [
    # architecture + extraction
    "ArchAdapter",
    "LlamaStyleFFN",
    "detect_adapter",
    "register_adapter",
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
    "feature_activations_at_layers",
    "active_features",
    "ActiveFeature",
    "lineage_score",
    "compare_scale",
    "ScaleCheck",
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
    "suppression_by_layer",
    "frontier_table",
    "rank_by_ablation_effect",
    "split_probes",
    "BatteryResult",
    "BatteryDiff",
    # tracing (direct vs total effects), from marv-hyena
    "capture_writes",
    "Writes",
    "decompose_logit",
    "Decomposition",
    "component",
    "all_components",
    "replace_outputs",
    "mean_writes",
    "mean_ablate",
    "logit_diff_metric",
    "patch_sweep",
    "PatchSweep",
    "trace_by_depth",
    # diagnostics: run before trusting an edit or a trace
    "health",
    "Health",
    "load_bearing",
    "write_norms",
    "find_bottlenecks",
    "dead_features",
    "null_model",
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
    # curated probe batteries
    "WORLD_CAPITALS",
    "capital_probes",
    "domain_probes",
    "broad_controls",
    "capital_edit_battery",
    "History",
    "Test",
    "BatteryTest",
    "changed_neurons",
    "revert_neurons",
    "history_callback",
    "checkpoint_loader",
]
