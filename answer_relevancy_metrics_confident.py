import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import evaluate

deepeval.login("CONFIDENT_AI_API_KEY")
answer_relevancy_metric = AnswerRelevancyMetric()
test_case1 = LLMTestCase(
    input="Who is the current president of the USA?",
    actual_output="Donald Trump",
    retrieval_context=["Joe Biden stays at the WhiteHouse"]
)

test_case2 = LLMTestCase(
    input="Who build the GPT 4.0 model?",
    actual_output="OpenAI",
    retrieval_context=["Open AI built the GPT models"]
)

evaluate(test_cases=[test_case1, test_case2], metrics=[answer_relevancy_metric])
