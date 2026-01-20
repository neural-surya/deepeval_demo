from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metrics = AnswerRelevancyMetric()
test_case = LLMTestCase(
    input="Who is the current president of the USA?",
    actual_output="Donald Trump",
    retrieval_context=["Joe Biden stays at the WhiteHouse"]
)

answer_relevancy_metrics.measure(test_case)
print(answer_relevancy_metrics.score)