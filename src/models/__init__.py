"""Model registry: name used on the command line -> class.

Each model chat adds its own entry, e.g. "dense": DenseNet.
"""
from src.models.cnn import CNN1D
from src.models.dense import DenseNet
from src.models.rnn import RNNClassifier
from src.models.transformer import TransformerClassifier
from src.models.trivial import LeadStatsLinear

MODELS = {
    "trivial": LeadStatsLinear,
    "dense": DenseNet,
    "cnn": CNN1D,
    "rnn": RNNClassifier,
    "transformer": TransformerClassifier,
}
