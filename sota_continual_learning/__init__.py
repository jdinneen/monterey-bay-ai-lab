"""
SOTA Continual Learning System.

State-of-the-art ML architecture designed for RTX 5090 with continuous learning capabilities.
"""

from .core import (
    ContinualLearner,
    DynamicMoE,
    ElasticWeightConsolidation,
    ExperienceReplayBuffer,
    SafetyMonitor
)
from .trainer import Trainer, PerformanceMonitor

__version__ = "0.1.0"
__all__ = [
    'ContinualLearner',
    'DynamicMoE',
    'ElasticWeightConsolidation',
    'ExperienceReplayBuffer',
    'SafetyMonitor',
    'Trainer',
    'PerformanceMonitor'
]
