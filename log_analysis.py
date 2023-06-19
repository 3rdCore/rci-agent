from pathlib import Path
from joblib import Memory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
import pandas as pd
import yaml
from llm_utils import count_tokens, compress_string, retry, yaml_parser
import tiktoken

# define your cachedir here
memory = Memory(location="/tmp/cache_error_analysis", verbose=0)


SYS_PROMPT = """You are helping to analyze the error of an agent trying to solve a task over a UI. 
The agent is recieving the task description and the html as input and he reflects on a possible plan to solve the task.
The agent then reflects on the plan and tries to execute it. 

If the interaction is too long, the middle part is removed and replaced
with a line containing ...

There might be a few possible reasons for failure:
pixel limited: since the agent doesn't see the UI, he may be missing key information to solve the task
selenium action limited: Actions are executed through selenium and maybe this interface prevents the agent from solving the task
api action limited: The agent is also submitting actions through a python api that is converting actions to selenium actions. Maybe this interface prevents the agent from solving the task.
plan limited: The plan is not good enough to solve the task, if the plan is not good because of e.g. pixel limited, then list pixel limited instead.
agent limited: The agent is not good enough to solve the task
other: There is another reason for failure and it will be discussed in the analysis.

Give a score for each possible failure reason from 1-10, where 1 is very unlikely and 10 is very likely. Answer in YAML format with nothing more, like this:

Analysis: "<free form detaild analysis of the agent's performance, don't add new line for valid yaml>"
Summary: "<one sentence summary of the analysis>"
Confidence Score: <how confident are you in your analysis on a 1-10 scale i.e. perhaps the logs don't contain enough information>
pixel limited: <1-10>
selenium action limited: <1-10>
api action limited: <1-10>
plan limited: <1-10>
agent limited: <1-10>
other: <1-10>
"""


def _truncate_prompt(prompt, allow_compression, max_tokens):
    """Just makes sure the prompt will fit in the API call. It can compress to remove redundant text or truncate the middle"""
    n_tokens = count_tokens(prompt)
    if allow_compression and n_tokens > max_tokens:
        compressed_prompt = compress_string(prompt)
        n_compressed_tokens = count_tokens(compressed_prompt)
        print(f"Compressed prompt from {n_tokens} to {n_compressed_tokens} tokens)")
        if n_compressed_tokens / n_tokens < 0.7:
            prompt = compressed_prompt
            n_tokens = n_compressed_tokens

    if n_tokens > max_tokens:
        print(
            f"Truncating prompt from {n_tokens} to {max_tokens} tokens by removing the middle"
        )
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(prompt)
        beginning = tokens[: max_tokens // 3]
        ending = tokens[-(max_tokens // 3) * 2 :]
        prompt = enc.decode(beginning) + "\n...\n" + enc.decode(ending)

    return prompt


@memory.cache
def error_analysis(
    logs,
    allow_compression=True,
    max_logs_tokens=6000,
    model_name="gpt-4",
    sys_prompt=SYS_PROMPT,
):
    """Given a log of an agent trying to solve a task, analyze the error and give a score for each possible failure reason. Returns a dict with the analysis."""
    logs = _truncate_prompt(logs, allow_compression, max_logs_tokens)
    chat = ChatOpenAI(
        model_name=model_name,
        temperature=0.2,
        openai_api_key="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",
    )
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=logs),
    ]

    ans_dict = retry(chat, messages, n_retry=2, parser=yaml_parser)

    return ans_dict


def analyse_all_logs(
    log_dict, allow_compression=True, max_tokens=6000, model_name="gpt-4"
):
    """Given a dict of task_name -> log, analyze the error for each log and return a dataframe with the results."""
    all_ans_dict = []

    keys = None

    for task_name, log in log_dict.items():
        ans_dict = error_analysis(log, allow_compression, max_tokens, model_name)
        ans_dict["task_name"] = task_name

        if keys is None:
            keys = set(ans_dict.keys())
        else:
            assert keys == set(ans_dict.keys())

        all_ans_dict.append(ans_dict)

    return pd.DataFrame(all_ans_dict)


if __name__ == "__main__":
    """Example usage. Run this file to see the output."""

    # logs_path = Path(ui_copilot.__file__).parent / "tests" / "data" / "logs.txt"
    logs_path = "/Users/rimassouel/rci-agent/history/chatgpt/enter-date/state_True-erci_1-irci_3/20230612-164029/few-shot/20230612-164309_fail.txt"
    with open(logs_path, "r") as f:
        logs = f.read()

    with get_openai_callback() as cb:
        answer = error_analysis(logs, max_logs_tokens=3000, model_name="gpt-3.5-turbo")
        print(yaml.dump(answer))

        print(cb)
