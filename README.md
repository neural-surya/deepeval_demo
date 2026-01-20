# DeepEval Metrics Examples

This project contains examples of using [DeepEval](https://github.com/confident-ai/deepeval) to evaluate LLM outputs. It demonstrates different metrics and usage patterns, including standalone measurements, using Golden datasets, and integrating with Confident AI.

## Files

### Answer Relevancy Metrics
*   `answer_relevancy_metrics_standalone.py`: Demonstrates how to use `AnswerRelevancyMetric` in a standalone manner to measure the score for a single test case.
*   `answer_relevancy_metrics_golden.py`: Shows how to use `Golden` datasets with `AnswerRelevancyMetric` and run an evaluation.
*   `answer_relevancy_metrics_confident.py`: Illustrates how to log in to Confident AI and run an evaluation with `AnswerRelevancyMetric` on multiple test cases.

### Contextual Precision Metrics
*   `contextual_precision_metrics_standalone.py`: Demonstrates how to use `ContextualPrecisionMetric` to measure the score and success for a single test case.

## Usage

Ensure you have `deepeval` installed and configured. You might need to set up your environment variables (e.g., OpenAI API key) as DeepEval relies on LLMs for evaluation.

```bash
pip install deepeval
```

Run the scripts using python:

```bash
python answer_relevancy_metrics_standalone.py
```
