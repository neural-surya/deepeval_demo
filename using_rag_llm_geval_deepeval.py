import using_rag_with_llm
from using_rag_with_llm import chain
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.dataset import EvaluationDataset

# Creating concise metrics
concise_metric = GEval(
    name = "Concise",
    criteria="Assess if actual output remains concise while preserving all the important information",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

# Creating completeness metric
completeness_metric = GEval(
    name = "Completeness",
    criteria="Assess if actual output retains all the key information from the input",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
)

test_case = LLMTestCase(
    input = "What are some of teh popular food of Ganjam?",
    actual_output = chain.invoke("What are some of teh popular food of Ganjam?"),
    expected_output= "Puri Upma, Achaar, Papad"
)

data_set = EvaluationDataset()
data_set.add_test_case(test_case)

evaluate(data_set.test_cases, metrics=[concise_metric, completeness_metric, AnswerRelevancyMetric()])
