
import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from langchain_classic.chains import RetrievalQA ##
import using_rag_with_llm
from typing import List, Any

test_data = [
    {
        "input": "What are some of teh popular food of Ganjam?",
        "expected_output": "Puri Upma, Achaar, Papad",
    },
    {
        "input": "Where is the best place to eat Puri Upma in Berhampur?",
        "expected_output": "In front of the Maa Kali temple near the new bus stand",
    },
    {
        "input": "What is the food item from Ganjam that is world famous?",
        "expected_output": "Achhar",
    }
]

#Creating Golden
goldens = []
for data in test_data:
    golden = Golden(
        input=data["input"],
        expected_output=data["expected_output"]
    )
    goldens.append(golden)

data_set = EvaluationDataset(goldens=goldens)
data_set.push(alias="data_set_Jan_23_2026")

data_set = EvaluationDataset()
data_set.pull(alias="data_set_Jan_23_2026")
qa_chain = RetrievalQA.from_chain_type(llm=using_rag_with_llm.llm, retriever=using_rag_with_llm.retriever)
def query_with_context(question):
    retrieved_document = using_rag_with_llm.context_retriever.invoke(question)
    response = qa_chain.invoke(question)
    return response, retrieved_document

def convert_golden_to_test_cases(data_set:EvaluationDataset) -> list[Any] | None:
    test_cases = []
    for golden in data_set.goldens:
        rag_response, context = query_with_context(golden.input)

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=rag_response['result'],
            expected_output=golden.expected_output,
            retrieval_context=[context]
        )
        test_cases.append(test_case)
    data_set.test_cases = test_cases
    return test_cases

test_cases_from_data_set = convert_golden_to_test_cases(data_set)

evaluate(test_cases_from_data_set, metrics=[
    AnswerRelevancyMetric(),
    FaithfulnessMetric(),
    ContextualPrecisionMetric(),
    ContextualRelevancyMetric()
])