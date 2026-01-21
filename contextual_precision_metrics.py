import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")
from deepeval.models import OllamaModel
from langchain_ollama import ChatOllama
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.5,
    max_token=250
)

model = OllamaModel(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434",
    temperature=0
)

contextual_precision_metric = ContextualPrecisionMetric()
test_case = LLMTestCase(
    input="What are the types of Bias an LLM can generate, give me just the heading",
    actual_output=llm.invoke("What are the types of Bias an LLM can generate").content,
    retrieval_context=["Gender Bias, Racial Bias, Ethnic Bias, Religious Bias, Political Bias, Cultural Bias, Educational Bias, Linguistic Bias, Ageism, Economic Bias, Nationalist Bias"],
    expected_output=""" 1. **Gender Bias**
                        2. **Racial Bias** 
                        3. **Ethnic Bias** 
                        4. **Religious Bias**
                        5. **Political Bias**
                        6. **Cultural Bias**
                        7. **Educational Bias**
                        8. **Linguistic Bias**
                        9. **Ageism**
                        10. **Economic Bias**
                        11. **Nationalist Bias**
                    """""
    )

evaluate(test_cases=[test_case], metrics = [contextual_precision_metric], identifier="local evaluation, local output, contextual precision")