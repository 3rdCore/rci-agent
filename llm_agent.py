import json
from prompt import Prompt
import time
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import difflib
from bs4 import BeautifulSoup

from pathlib import Path
from selenium.webdriver.common.keys import Keys
import os
import logging
import tiktoken

from computergym.miniwob.miniwob_interface.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
import re


# using tiktoken tokenizer to evaluate the number of token sent and received from OpenAI api
def count_tokens(text, model):
    model_map = {
        "chatgpt": "gpt-3.5-turbo",
        "gpt4": "gpt-4",
        "davinci": "text-davinci-003",
        "ada": "ada",
        "babbage": "babbage",
        "curie": "curie",
        "davinci1": "davinci",
        "davinci2": "text-davinci-002",
    }

    enc = tiktoken.encoding_for_model(model_map.get(model))
    return len(enc.encode(text))


def get_html_code(input):
    """
    Parses the given input string to, retrieve the HTML code block using BeautifulSoup.
    When extracting the html code from the input string, BS also rearrange the tag attributes order, which is an undesired behavior that makes it difficult to replace the html code with rearranged attributes from the non-modified input string. For this reason, we overwrite the input string with the BS-processed input string.
    Args:
        input (str): The input string to parse.

    Returns:
        A tuple containing two strings:
        - The input string with any HTML code replaced by a placeholder string.
        - The largest HTML tag found in the input string, as a string.
    """

    # Parse the HTML code using BeautifulSoup
    soup = BeautifulSoup(input, "html.parser")
    # Find all HTML tags in the soup
    tags = soup.find_all()

    # Find the largest HTML tag
    largest_tag = None
    largest_length = 0

    for tag in tags:
        length = len(str(tag))
        if length > largest_length:
            largest_length = length
            largest_tag = tag

    # Print the largest HTML tag
    return (str(soup), str(largest_tag))


class TextFileWriter:
    """A class for writing text to a log_file with HTML code and reccurent paragraph detection and replacement.
    Attributes:
    file_path (str): The path to the file to write to.
    prompt (Prompt): The Prompt object used to retrieve the human-crafted prompts.

    Methods:
        write(pt): Writes the given text to the file, replacing any recurrent paragraphs in the reverse_dict with their according placeholder.
        Also detects and replaces the full HTML code with the placeholder [HTML_CODE] and the difference between the previous and the current HTML in the github PR format.

        write_explanation(): Writes an explanation of the Prompt object and the reverse_dict to the file, followed by the existing content of the file.
    """

    def __init__(self, file_path, prompt):
        """
        Initializes a TextFileWriter object with the given file path and Prompt object.

        Args:
            file_path (str): The path to the file to write to.
            prompt (Prompt): The Prompt object used to generate the text to write.
        """
        self.file_path = file_path
        self.reverse_dict = prompt.get_reverse_dict()
        self.html_state = ""

    def write(self, pt):
        """
        Writes the given text to the log file, replacing any recurrent paragraphs in the reverse_dict with their according placeholder.
        Also detects and replaces the full HTML code with the placeholder [HTML_CODE] and the difference between the previous and the current HTML in the github PR format.

        Args:
            pt (str): The text to write to the file.
        """
        # retrieve value if pt in self.reverse_dict
        for key in self.reverse_dict.keys():
            if key in pt:
                pt = pt.replace(key, self.reverse_dict[key])

        pt, match = get_html_code(pt)
        if match != "None":
            if len(self.html_state) > 0:
                diff = difflib.unified_diff(
                    self.html_state.splitlines(),
                    match.splitlines(),
                    lineterm="",
                    fromfile="old",
                    tofile="new",
                )
                diff_str = "\n".join(list(diff))
                if len(diff_str) > 0:
                    pt = pt.replace(
                        match,
                        "[HTML CODE]"
                        + "\n"
                        + "Here is the difference between the previous HTML state and the current HTML state: \n"
                        + diff_str,
                    )
                else:
                    pt = pt.replace(match, "[HTML CODE]")

            self.html_state = match

        with open(self.file_path, "a") as f:
            f.write(pt)

    def write_explanation(self):
        """
        Writes the explanation prompt and the reverse_dict at the beggining of the the file,
        followed by the existing content of the file.
        """
        with open("logging_prompt.txt", "r") as f:
            explanation_prompt = f.read()

        # open self.file_path and and text from the top :
        with open(self.file_path, "r") as f:
            existing_content = f.read()

        with open(self.file_path, "w") as f:
            f.write(explanation_prompt)
            json.dump(self.reverse_dict, f)
            f.write("\n")
            f.write(existing_content)


class LLMAgent:
    def __init__(
        self,
        env: str,
        rci_plan_loop: int = 1,
        rci_limit: int = 1,
        llm="chatgpt",
        with_task=True,
        state_grounding=True,
        exp_path="",
    ) -> None:
        self.rci_limit = rci_limit
        self.rci_plan_loop = rci_plan_loop
        self.llm = llm
        self.prompt = Prompt(env=env)
        self.state_grounding = state_grounding
        self.api_key = ""
        self.load_model()
        self.number_of_token_sent = 0
        self.number_of_token_received = 0
        self.number_of_calls = 0
        self.html_state = ""
        self.task = ""
        self.with_task = with_task
        self.current_plan = ""
        self.past_plan = []
        self.past_instruction = []
        self.custom_gaol = False
        self.exp_path = exp_path
        self.history_name = time.strftime("%Y%m%d-%H%M%S")

        if self.prompt.example_prompt:
            self.file_path = Path(f"{self.exp_path}/few-shot/{self.history_name}.txt")
        else:
            self.file_path = Path(f"{self.exp_path}/zero-shot/{self.history_name}.txt")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            str(self.file_path.parent),
            f"{self.history_name}_compressed.txt",
        )
        self.writer = TextFileWriter(filename, self.prompt)

    def load_model(self):
        """
        Loads the model and the API key from the config.json file.
        """
        with open("config.json") as config_file:
            api_key = json.load(config_file)["api_key"]
            self.api_key = api_key
        if self.llm == "chatgpt":
            self.model = "gpt-3.5-turbo"
        elif self.llm == "gpt4":
            self.model = "gpt-4"
        elif self.llm == "davinci":
            self.model = "text-davinci-003"
        elif self.llm == "ada":
            self.model = "ada"
        elif self.llm == "babbage":
            self.model = "babbage"
        elif self.llm == "curie":
            self.model = "curie"
        elif self.llm == "davinci1":
            self.model = "davinci"
        elif self.llm == "davinci2":
            self.model = "text-davinci-002"
        else:
            raise NotImplemented

    def save_task(self):
        """
        Saves the task in the log file.
        """
        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "INPUT TASK" + "-" * 30 + "\n"
            pt += self.task
            f.write(pt)

        self.writer.write(pt)

    def save_result(self, result, cause=""):
        with open(self.file_path, "a") as f:
            if result:
                pt = "\n\nSUCCESS : reward was obtained\n\n"
                f.write(pt)
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_success.txt"
                )
                self.writer.write(pt)

            else:
                pt = "\n\nFAIL : reward was not obtained."
                if cause != "":
                    pt += " reason :" + str(cause)
                pt += "\n\n"
                f.write(pt)
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_fail.txt"
                )
                self.writer.write(pt)

        os.rename(self.file_path, new_file_path)

        return

    def save_message(self, pt):
        """
        Saves the message (input of the LLM) in the log file.
        """
        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "INPUT" + "-" * 30 + "\n" + pt + "\n"
            f.write(pt)

        self.writer.write(pt)

        return

    def save_response(self, response):
        """
        Saves the response (output of the LLM) in the log file.
        """
        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "OUTPUT" + "-" * 30 + "\n" + response + "\n"
            f.write(pt)

        self.writer.write(pt)

        return

    def save_error(self, response):
        """
        Saves the error (output of the LLM) in the log file in case the main thread of the benchmark crashes.
        """
        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "INPUT" + "-" * 30 + "\n" + response + "\n"
            f.write(pt)

            new_file_path = self.file_path.with_name(f"{self.history_name}_error.txt")
            os.rename(self.file_path, new_file_path)

        self.writer.write(pt)

        return

    def save_action(self, response):
        """
        Saves the selenium-compatible action (processed output of the LLM) executed on the environment in the log file.
        """
        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "ACTION" + "-" * 30 + "\n" + response + "\n"
            f.write(pt)

        self.writer.write(pt)

        return

    def save_logging(self, response):

        with open(self.file_path, "a") as f:
            pt = "\n" + "-" * 30 + "LOGGING" + "-" * 30 + "\n" + response + "\n"
            f.write(pt)

    def set_goal(self, goal: str):
        self.custom_gaol = True
        self.task = goal
        self.save_task()
        return

    def instruction_history_prompt(self):
        pt = "\n\n"
        pt += "We have a history of instructions that have been already executed by the autonomous agent so far.\n"
        self.writer.reverse_dict[pt] = "[HISTORY]"
        if not self.past_instruction:
            pt += "No instruction has been executed yet."
        else:
            for idx, inst in enumerate(self.past_instruction):
                pt += f"{idx+1}: "
                pt += inst
                pt += "\n"
        pt += "\n\n"

        return pt

    def webpage_state_prompt(self, init_plan: bool = False, with_task=False):
        pt = "\n\n"
        pt += "Below is the HTML code of the webpage where the agent should solve a task.\n"
        self.writer.reverse_dict[pt] = "[HTML CODE BELOW]"
        pt += self.html_state
        pt += "\n\n"
        if self.prompt.example_prompt and (init_plan or self.rci_plan_loop == -1):
            pt += self.prompt.example_prompt
            pt += "\n\n"
        if with_task:
            pt += "Current task: "
            pt += self.task
            pt += "\n"

        return pt

    def update_html_state(self, state: str):
        self.html_state = state

        return

    def rci_plan(self, pt=None):
        pt += "\n\nFind problems with this plan for the given task compared to the example plans.\n\n"
        self.save_message(pt)
        self.writer.reverse_dict[
            "Find problems with this plan for the given task compared to the example plans."
        ] = "[FIND PROBLEMS]"
        criticizm = self.get_response(pt)
        self.save_response(criticizm)
        pt += criticizm

        pt += "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
        # pt += self.webpage_state_prompt()
        self.save_message(pt)
        self.writer.reverse_dict[
            "Based on this, what is the plan for the agent to complete the task?"
        ] = "[WHAT IS THE PLAN]"

        plan = self.get_response(pt)
        self.save_response(plan)
        return pt, plan

    def rci_action(self, instruciton: str, pt=None):
        instruciton = self.process_instruction(instruciton)

        loop_num = 0
        while self.check_regex(instruciton):
            logging.info(
                f"instruciton not valid, RCI_action loop number : {loop_num + 1}"
            )
            self.save_logging(f"instruciton not valid, RCI_action loop number : {loop_num + 1}")

            if loop_num >= self.rci_limit:
                logging.error("Too many attemps to get a valid instruction : RCI failed")
                raise ValueError("Too many attemps to get a valid instruction : RCI failed")

            pt += self.prompt.rci_action_prompt
            self.save_message(pt)
            instruciton = self.get_response(pt)
            self.save_response(instruciton)

            pt += instruciton
            instruciton = self.process_instruction(instruciton)

            loop_num += 1

        return pt, instruciton

    def check_regex(self, instruciton):
        return (
            (not re.search(self.prompt.clickxpath_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.chatgpt_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.davinci_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.press_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.clickoption_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.movemouse_regex, instruciton, flags=re.I))
        )

    def process_instruction(self, instruciton: str):
        end_idx = instruciton.find("`")
        if end_idx != -1:
            instruciton = instruciton[:end_idx]

        instruciton = instruciton.replace("`", "")
        instruciton = instruciton.replace("\n", "")
        instruciton = instruciton.replace("\\n", "\n")
        instruciton = instruciton.strip()
        instruciton = instruciton.strip("'")

        return instruciton

    def get_plan_step(self):
        idx = 1
        while True:
            if (str(idx) + ".") not in self.current_plan:
                return (idx - 1) + 1
            idx += 1

    def initialize_plan(self):
        if not self.custom_gaol:
            if self.with_task:
                self.initialize_task()

        if not self.prompt.init_plan_prompt or self.rci_plan_loop == -1:
            return

        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(True, with_task=self.with_task)
        pt += self.prompt.init_plan_prompt

        self.save_message(pt)
        try:
            message = "\n" + self.get_response(pt)
            self.writer.reverse_dict[message] = "[INIT PLAN]"
            self.save_response(message)
        except Exception as e:
            self.save_error(str(e))
            raise e

        pt += message

        for _ in range(self.rci_plan_loop):
            pt, message = self.rci_plan(pt)
            pt += message

        self.current_plan = message
        self.writer.reverse_dict[self.current_plan] = "[CURRENT PLAN]"
        return

    def get_response(self, pt):
        import inspect

        logging.info(
            f"Send a request to the language model from {inspect.stack()[1].function}"
        )
        self.save_logging(f"Send a request to the language model from {inspect.stack()[1].function}")

        # increment number of calls to the API
        self.number_of_calls += 1
        # store number of tokens sent to the API
        self.number_of_token_sent += count_tokens(pt, model=self.llm)

        while True:  # loop until we get a response from the API
            try:
                time.sleep(1)

                params = {"temperature": 0, "max_tokens": 256}
                open_ai_params = {
                    "top_p": 1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }

                openai = ChatOpenAI(
                    model_name=self.model,
                    **params,
                    model_kwargs=open_ai_params,
                    openai_api_key=self.api_key,
                )

                messages = [
                    SystemMessage(
                        content="You are an autoregressive language model that completes user's sentences. You should not conversate with user."
                    ),
                    HumanMessage(content=pt),
                ]

                response = openai(messages).content

            except Exception as e:
                logging.error("OpenAI related error :", str(e))
                if "maximum context" in str(e):
                    raise ValueError(
                        str(e)
                    )  # This is the only case for which we want to raise an error otherwise the model just loop until OpenAI respond.
                time.sleep(10)
            else:
                if response:
                    break

        self.number_of_token_received += count_tokens(response, model=self.llm)
        return response

    def generate_action(self) -> str:
        pt = self.prompt.base_prompt
        pt += self.webpage_state_prompt(with_task=self.with_task)
        if self.prompt.init_plan_prompt and self.rci_plan_loop != -1:
            pt += self.current_plan_prompt()
        pt += self.instruction_history_prompt()
        if self.past_instruction:
            update_action_prompt = self.prompt.action_prompt.replace(
                "{prev_inst}", self.past_instruction[-1]
            )
            if len(self.past_instruction) == 1:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "2nd"
                )
            elif len(self.past_instruction) == 2:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "3rd"
                )
            else:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", f"{len(self.past_instruction)+1}th"
                )

            action_prompt = update_action_prompt
        else:
            action_prompt = self.prompt.first_action_prompt

        if self.rci_plan_loop == -1:
            action_prompt = "Based on the task, " + action_prompt
        else:
            action_prompt = (
                "Based on the plan and the history of instructions executed so far, "
                + action_prompt
            )

        pt += action_prompt

        self.save_message(pt)
        message = self.get_response(pt)
        self.save_response(message)

        pt += self.process_instruction(message) + "`."

        pt, message = self.update_action(pt, message)

        pt, instruction = self.rci_action(pt=pt, instruciton=message)

        self.past_instruction.append(instruction)
        return instruction

    def update_action(self, pt=None, message=None):
        if self.prompt.update_action and self.state_grounding:
            pt += self.prompt.update_action

            self.save_message(pt)
            message = self.get_response(pt)
            self.save_response(message)
            pt += message

        return pt, message

    def current_plan_prompt(self):
        pt = "\n\n"
        pt += "Here is a plan you are following now.\n"
        pt += f"{self.current_plan}"
        pt += "\n\n"

        return pt

    def convert_to_miniwob_action(self, instruction: str):
        instruction = instruction.split(" ")
        inst_type = instruction[0]
        inst_type = inst_type.lower()

        if inst_type == "type":
            characters = " ".join(instruction[1:])
            characters = characters.replace('"', "")
            return MiniWoBType(characters)
        elif inst_type == "clickid":
            element_id = " ".join(instruction[1:])
            return MiniWoBElementClickId(element_id)
        elif inst_type == "press":
            key_type = instruction[1].lower()
            if key_type == "enter":
                return MiniWoBType("\n")
            elif key_type == "space":
                return MiniWoBType(" ")
            elif key_type == "arrowleft":
                return MiniWoBType(Keys.LEFT)
            elif key_type == "arrowright":
                return MiniWoBType(Keys.RIGHT)
            elif key_type == "backspace":
                return MiniWoBType(Keys.BACKSPACE)
            elif key_type == "arrowup":
                return MiniWoBType(Keys.UP)
            elif key_type == "arrowdown":
                return MiniWoBType(Keys.DOWN)
            else:
                logging.error(f"Invalid key type : {key_type}")
                raise NotImplemented
        elif inst_type == "movemouse":
            xpath = " ".join(instruction[1:])
            return MiniWoBMoveXpath(xpath)
        elif inst_type == "clickxpath":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickXpath(xpath)
        elif inst_type == "clickoption":
            xpath = " ".join(instruction[1:])
            return MiniWoBElementClickOption(xpath)
        else:
            logging.error(f"Invalid instruction type : {inst_type}")
            raise ValueError("Invalid instruction")
