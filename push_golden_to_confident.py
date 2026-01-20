# This file gives example of constructing golden from raw data(here json), adding golden to dataset and pushing them to Confident AI
import json

from deepeval.dataset import EvaluationDataset, Golden

data = json.load(open('data.json'))
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
print(data_set)
data_set.push(alias="DataSet_Jan_20_26")
