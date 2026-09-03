"""CASCADE model components: CCMP backbone, propagation head, full model.

The backbone needs torch. Imports are guarded so ``import cel.models``
still succeeds in a torch-free environment (for example the lightweight CI
image used for the data and features layers); the symbols simply are not
exported there.
"""

try:
    from .cascade import CASCADE
    from .ccmp_backbone import CCMPBackbone, ConceptStream
    from .concept_gnn import RNAFM_CONCEPT, ConceptGNN
    from .offtarget_predictor import OffTargetPredictor, assert_leakage_free
    from .prop_layer import MonotoneEquilibriumPropagation, analytic_propagate
    from .propagation_head import PropagationHead
    from .rc_equivariant import RCEquivariantLinear

    __all__ = [
        "CASCADE",
        "CCMPBackbone",
        "ConceptGNN",
        "ConceptStream",
        "MonotoneEquilibriumPropagation",
        "OffTargetPredictor",
        "PropagationHead",
        "RCEquivariantLinear",
        "RNAFM_CONCEPT",
        "analytic_propagate",
        "assert_leakage_free",
    ]
except ImportError:  # torch not installed
    __all__ = []
