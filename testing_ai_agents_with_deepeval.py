import os
import deepeval
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain.tools import tool
from deepeval.test_case import ToolCall
from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
from langchain_classic import hub
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric

os.environ["USER_AGENT"] = "MyLangChainApp/1.0 (colimop977@mustaer.com)"

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.5
)
search_tool = DuckDuckGoSearchRun()

@tool
def add_numbers(a:int, b:int) -> int:
    """Add two numbers and return results."""
    return int(a) + int(b)

@tool
def subtract_numbers(a:int, b:int) -> int:
    """Subtract two numbers and return results."""
    return int(a) - int(b)

tools = [search_tool, add_numbers, subtract_numbers]
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def query_ai_agent(question):
    responses = agent_executor.invoke({"input": question})
    # Iterate to find the first valid tool that isn't an exception
    for action, result in responses['intermediate_steps']:
        if action.tool != "_Exception":
            return responses, action.tool, action.tool_input

    # Fallback if only exceptions exist
    action, _ = responses['intermediate_steps'][0]
    return responses, action.tool, action.tool_input

test_data = [
    {
        "input": "What is the sum of 20 and 40",
        "expected_output": "60",
        "tool_called": [
            ToolCall(name="add_numbers"),
        ]
    },
    {
        "input": "Who is the Chief Minister of Odisha in 2026, just give me the name",
        "expected_output": "Mohan Majhi",
        "tool_called": [
            ToolCall(name="duckduckgo_search"),
        ]
    },
]

test_cases = []
for data in test_data:
    response, tool, tool_input = query_ai_agent(data["input"])
    test_case = LLMTestCase(
        input=data["input"],
        expected_tools=data['tool_called'],
        tools_called=[ToolCall(name=tool)],
        actual_output=response['output'],
        expected_output=data['expected_output']
    )
    test_cases.append(test_case)

metrics = ToolCorrectnessMetric()

for test_case in test_cases:
    metrics.measure(test_case=test_case)
    print("Metrics Score: ", metrics.score, "\nMetrics Reason: \n", metrics.reason, "\nMetrics Successful: ", metrics.is_successful())

deepeval.evaluate(
    test_cases,
    metrics=[ToolCorrectnessMetric()],
)
