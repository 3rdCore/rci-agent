import json
from prompt import Prompt
import time
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

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
        "davinci2": "text-davinci-002"
    }

    enc = tiktoken.encoding_for_model(model_map.get(model))
    return len(enc.encode(text))

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
        self.cause = ""
        self.exp_path = exp_path

        self.history_name = time.strftime("%Y%m%d-%H%M%S")

        if self.prompt.example_prompt:
            self.file_path = Path(
                f"{self.exp_path}/few-shot/{self.history_name}.txt"
            )
        else:
            self.file_path = Path(
                f"{self.exp_path}/zero-shot/{self.history_name}.txt"
            )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def load_model(self):
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
        with open(self.file_path, "a") as f:
            f.write("\n")
            ho_line = "-" * 30 + "INPUT TASK" + "-" * 30 
            f.write(ho_line)
            f.write("\n")
            f.write(self.task)

    def save_result(self, result, cause =""):
        with open(self.file_path, "a") as f:
            if result:
                f.write("\n\nSUCCESS\n\n")
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_success.txt"
                )
            else:
                f.write("\n\nFAIL : "+ cause +"\n\n")
                new_file_path = self.file_path.with_name(
                    f"{self.history_name}_fail.txt"
                )

        os.rename(self.file_path, new_file_path)

        return

    def save_message(self, pt):
        with open(self.file_path, "a") as f:
            f.write("\n")
            ho_line = "-" * 30 + "INPUT" + "-" * 30 
            f.write(ho_line)
            f.write("\n")
            f.write(pt)
            f.write("\n")

        return

    def save_response(self, response):
        with open(self.file_path, "a") as f:
            f.write("\n")
            ho_line = "-" * 30 + "OUTPUT" + "-" * 30 
            f.write(ho_line)
            f.write("\n")
            f.write(response)
            f.write("\n")

        return

    def set_goal(self, goal: str):
        self.custom_gaol = True
        self.task = goal
        self.save_task()
        return

    def instruction_history_prompt(self):
        pt = "\n\n"
        pt += "We have a history of instructions that have been already executed by the autonomous agent so far.\n"
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
        criticizm = self.get_response(pt)
        self.save_response(criticizm)
        pt += criticizm

        pt += "\n\nBased on this, what is the plan for the agent to complete the task?\n\n"
        # pt += self.webpage_state_prompt()
        self.save_message(pt)
        plan = self.get_response(pt)
        self.save_response(plan)
        return pt, plan

    def rci_action(self, instruciton: str, pt=None):
        instruciton = self.process_instruction(instruciton)

        loop_num = 0
        while self.check_regex(instruciton):
            if loop_num >= self.rci_limit:
                logging.error(instruciton, "Action RCI failed")
                raise ValueError("Action RCI failed")

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
        message = "\n" + self.get_response(pt)
        self.save_response(message)

        pt += message

        for _ in range(self.rci_plan_loop):
            pt, message = self.rci_plan(pt)
            pt += message

        self.current_plan = message

        return

    def get_response(self, pt):
        import inspect

        logging.info(
            f"Send a request to the language model from {inspect.stack()[1].function}"
        )
        #increment number of calls to the API
        self.number_of_calls += 1
        #store number of tokens sent to the API
        self.number_of_token_sent += count_tokens(pt, model=self.llm)

        while True:
            try:
                time.sleep(1)

                params = {
                    "temperature": 0,
                    "max_tokens": 256
                }
                open_ai_params = {
                    "top_p": 1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
                
                openai = ChatOpenAI(model_name="gpt-3.5-turbo", **params, model_kwargs=open_ai_params, openai_api_key = self.api_key)

                messages = [
                    SystemMessage(content = "You are an autoregressive language model that completes user's sentences. You should not conversate with user."),
                    HumanMessage(content = pt)
                ]

                response = openai(messages).content

            except Exception as e:
                print(e)
                if "maximum context" in str(e):
                    raise ValueError
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
            raise ValueError("Invalid instruction")
            