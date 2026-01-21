import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.5,
    max_token=250
)

# testing LLM with actual output received from LLM call
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import evaluate

answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
    input = "What is the capital of Odisha?",
    actual_output=llm.invoke("What is the capital of Odisha?").content,
    retrieval_context=["The capital of Odisha is also called Temple city"]
)

evaluate(test_cases=[test_case], metrics=[answer_relevancy_metric])