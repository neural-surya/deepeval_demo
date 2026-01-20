from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

answer_relevancy_metric = AnswerRelevancyMetric()
golden = Golden(
    input="Who is the current president of the USA?",
    expected_output="Joe Biden",
    context=["Joe Biden stays at the WhiteHouse"]
)

# Creating Golden and adding it to the dataset
dataset = EvaluationDataset()
dataset.add_golden(golden)
print("Before adding test cases to dataset: \n",dataset)


# Creating test cases from Golden and adding test cases to the dataset
for golden in dataset.goldens:
    test_case = LLMTestCase(
        input = golden.input,
        actual_output='Joe Biden',
        expected_output = golden.expected_output,
        retrieval_context = golden.context
    )
    dataset.add_test_case(test_case)
print("After adding test cases to dataset: \n",dataset)

evaluate(test_cases=dataset.test_cases, metrics=[answer_relevancy_metric])