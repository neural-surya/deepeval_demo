# This example shows how to pull the golden from dataset from confident ai and prepare test cases
def mock_llm_app(input):
    match input:
        case "1":
            return "Joe Biden"
        case "2":
            return "OpenAI"
    return None

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric


cloud_dataset = EvaluationDataset()
cloud_dataset.pull(alias="DataSet_Jan_20_26")
test_cases = []
for i, golden in enumerate(cloud_dataset.goldens):
    test_case = LLMTestCase(
        input = golden.input,
        actual_output = mock_llm_app(str(i+1)),
        expected_output = golden.expected_output,
    )
    cloud_dataset.add_test_case(test_case)

# Run evaluate
evaluate(test_cases=cloud_dataset.test_cases, metrics=[AnswerRelevancyMetric()])

