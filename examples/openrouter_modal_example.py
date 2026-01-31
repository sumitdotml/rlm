"""
Example usage of RLM with Modal sandbox and OpenRouter.

This demonstrates the full RLM loop:
1. RLM receives a prompt
2. LLM generates Python code to solve the problem
3. Code executes in Modal sandbox (can call llm_query for sub-LM calls)
4. Iterates until FINAL(...) answer is found

Run with: uv run python -m examples.openrouter_modal_example
"""

import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")

    print("=" * 60)
    print("RLM + Modal + OpenRouter Example")
    print("=" * 60)

    # Create RLM with Modal environment and OpenRouter backend
    logger = RLMLogger(log_dir="./logs")
    rlm = RLM(
        backend="openrouter",
        backend_kwargs={
            "api_key": api_key,
            "model_name": "arcee-ai/trinity-large-preview:free",
        },
        environment="modal",
        environment_kwargs={
            "app_name": "rlm-openrouter-example",
        },
        max_iterations=10,
        verbose=True,
        logger=logger,
    )

    # Question that benefits from code execution and reasoning
    prompt = """
    Analyze the bias-variance tradeoff in machine learning.

    Write code to:
    1. Generate synthetic polynomial regression data with noise
    2. Fit models of increasing complexity (degree 1, 5, 15 polynomials)
    3. Calculate and compare training vs test MSE for each
    4. Use llm_query to interpret the results and explain what's happening

    Return a clear explanation of bias-variance tradeoff based on your empirical results.
    """

    print("\n[Prompt]")
    print("-" * 40)
    print(prompt.strip())
    print("-" * 40)

    result = rlm.completion(prompt)

    print("\n[Final Answer]")
    print("=" * 60)
    print(result.response)
    print("=" * 60)

    print(f"\nExecution time: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")


if __name__ == "__main__":
    main()
