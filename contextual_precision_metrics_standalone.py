from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

contextual_precision_metrics = ContextualPrecisionMetric()
test_case = LLMTestCase(
    input="Who is the current president of the USA?",
    expected_output="Donald J Trump is the current president of the United States",
    actual_output="Sulivan Harris",
    retrieval_context=["Donal Trump stays at the WhiteHouse"]
)
contextual_precision_metrics.measure(test_case)
print(contextual_precision_metrics.score)
print(contextual_precision_metrics.success)