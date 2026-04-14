from .base_strategy import BaseStrategy
from .random_sampling import RandomSampling
from .entropy_sampling import EntropySampling
from .bald import BALDSampling
from .badge import BADGESampling
from .coreset import CoreSetSampling
from .ee_uncertainty import EEUncertainty
from .mask_guided_coreset import MaskGuidedCoreSet
from .ee_al import EEActiveLearning

STRATEGY_REGISTRY = {
    'random':            RandomSampling,
    'entropy':           EntropySampling,
    'bald':              BALDSampling,
    'badge':             BADGESampling,
    'coreset':           CoreSetSampling,
    'ee_uncertainty':    EEUncertainty,
    'mask_guided_coreset': MaskGuidedCoreSet,
    'ee_al':             EEActiveLearning,   # ← OUR METHOD
}


def build_strategy(name: str, model, idxs_lb, idxs_unlb, cfg, device) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](model, idxs_lb, idxs_unlb, cfg, device)
