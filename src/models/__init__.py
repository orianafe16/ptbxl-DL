"""Model registry: name used on the command line -> class.

Each model chat adds its own entry, e.g. "dense": DenseNet.
"""
from src.models.dense import DenseNet
from src.models.trivial import LeadStatsLinear

MODELS = {
    "trivial": LeadStatsLinear,
    "dense": DenseNet,
}
