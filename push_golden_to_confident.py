# This file gives example of constructing golden from raw data(here json), adding golden to dataset and pushing them to Confident AI
import json
from typing import override

from deepeval.dataset import EvaluationDataset, Golden

data = json.load(open('data/data.json'))
test_data = data['test_data']

# Construct Golden
golden_ds = []
for item in test_data:
    golden = Golden(
        input = item['input'],
        expected_output = item['expected_output'],
    )
    golden_ds.append(golden)

# Construct the dataset
data_set = EvaluationDataset(goldens=golden_ds)
data_set.delete(alias="DataSet_Jan_20_26")
data_set.push(alias="DataSet_Jan_20_26")

