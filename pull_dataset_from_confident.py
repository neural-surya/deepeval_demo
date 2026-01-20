from deepeval.dataset import EvaluationDataset, Golden
from pprint import pprint

# Pulling dataset from Confident AI
pulled_dataset = EvaluationDataset()
pulled_dataset.pull(alias="DataSet_Jan_20_26")
pprint(pulled_dataset)