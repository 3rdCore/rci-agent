import collections
import json
from langchain.schema import HumanMessage
from warnings import warn
import tiktoken
import re

import yaml


def retry(chat, messages, n_retry, parser):
    """Retry querying the chat models until it returns a valid value with a maximum number of retries.

    Args:
        chat (function): a langchain chat taking a list of messages and returning the llm's message.
        messages (list): the list of messages so far.
        n_retry (int): the maximum number of retries.
        parser (function): a function taking a message and returning a tuple (value, valid, retry_message)
          where value is the parsed value, valid is a boolean indicating if the value is valid and retry_message
          is a message to display to the user if the value is not valid.
    """

    for i in range(n_retry):
        answer = chat(messages)
        value, valid, retry_message = parser(answer.content)
        if valid:
            return value

        msg = f"Query faild. Retrying {i+1}/{n_retry}.\n[LLM]:\n{answer.content}\n[User]:\n{retry_message}"
        warn(msg)
        messages.append(answer)
        messages.append(HumanMessage(content=retry_message))

    raise ValueError(f"Could not parse a valid value after {n_retry} retries.")


def truncate_tokens(text, max_tokens=8000, start=0, model_name="gpt-4"):
    """Use tiktoken to truncate a text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) - start > max_tokens:
        return enc.decode(tokens[start : (start + max_tokens)])
    else:
        return text


def count_tokens(text, model="gpt-4"):
    """Use tiktoken to count the number of tokens in a text."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def json_parser(message):
    """Parse a json message for the retry function."""

    try:
        value = json.loads(message)
        valid = True
        retry_message = ""
    except json.JSONDecodeError as e:
        warn(e)
        value = {}
        valid = False
        retry_message = "Your response is not a valid json. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def yaml_parser(message):
    """Parse a yaml message for the retry function."""

    # saves gpt-3.5 from some yaml parsing errors
    message = re.sub(r":\s*\n(?=\S|\n)", ": ", message)

    try:
        value = yaml.safe_load(message)
        valid = True
        retry_message = ""
    except yaml.YAMLError as e:
        warn(str(e))
        value = {}
        valid = False
        retry_message = "Your response is not a valid yaml. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def _compress_chunks(text, identifier, skip_list, split_regex="\n\n+"):
    """Compress a string by replacing redundant chunks by identifiers. Chunks are defined by the split_regex."""
    text_list = re.split(split_regex, text)
    text_list = [chunk.strip() for chunk in text_list]
    counter = collections.Counter(text_list)
    def_dict = {}
    id = 0

    # Store items that occur more than once in a dictionary
    for item, count in counter.items():
        if count > 1 and item not in skip_list and len(item) > 10:
            def_dict[f"{identifier}-{id}"] = item
            id += 1

    # Replace redundant items with their identifiers in the text
    compressed_text = "\n".join(text_list)
    for key, value in def_dict.items():
        compressed_text = compressed_text.replace(value, key)

    return def_dict, compressed_text


def compress_string(text):
    """Compress a string by replacing redundant paragraphs and lines with identifiers."""

    # Perform paragraph-level compression
    def_dict, compressed_text = _compress_chunks(
        text, identifier="§", skip_list=[], split_regex="\n\n+"
    )

    # Perform line-level compression, skipping any paragraph identifiers
    line_dict, compressed_text = _compress_chunks(
        compressed_text, "¶", list(def_dict.keys()), split_regex="\n+"
    )
    def_dict.update(line_dict)

    # Create a definitions section
    def_lines = ["<definitions>"]
    for key, value in def_dict.items():
        def_lines.append(f"{key}:\n{value}")
    def_lines.append("</definitions>")
    definitions = "\n".join(def_lines)

    return definitions + "\n" + compressed_text
