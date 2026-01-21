import deepeval
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
deepeval.login("CONFIDENT_AI_API_KEY")

model = OllamaModel(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434",
    temperature=0
)

# Here instead of relying on cloud version of LLM we used local LLM(deepseek here for example)
answer_relevancy_metric = AnswerRelevancyMetric(model=model)

test_case1 = LLMTestCase(
    input="Who is the current president of the USA?",
    actual_output="Donald Trump",
    retrieval_context=["Joe Biden stays at the WhiteHouse"]
)

evaluate(test_cases=[test_case1], metrics=[answer_relevancy_metric])