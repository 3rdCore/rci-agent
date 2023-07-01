from langchain import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import time
import tiktoken
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage

params = {"temperature": 0, "max_tokens": 256}
open_ai_params = {
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

openai = ChatOpenAI(
    model_name="gpt-4",
    **params,
    model_kwargs=open_ai_params,
    openai_api_key="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",
)


template="You are an autoregressive language model that completes user's sentences. You should not conversate with user."
input_prompt = "Write a python code to print hello world"

messages = [
    SystemMessage(
        content=template
    ),
    HumanMessage(content=input_prompt),
]

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(input_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
messages == chat_prompt.format_prompt().to_messages()
openai_chain = LLMChain(prompt=chat_prompt, llm=openai)
start_time = time.time()
response =openai_chain.run({})
end_time = time.time()
inference_speed = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(response)) / (end_time - start_time)
print(f"Inference speed 4 GPUs: {inference_speed:.2f} tokens/second")
#openai_chain = LLMChain(prompt=chat_prompt, llm=openai)
#print(openai_chain.run({}))

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/starchat-beta",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 256, "device_map" : "auto"},
    )
llm_chain = LLMChain(prompt=chat_prompt, llm=llm)

start_time = time.time()
response =llm_chain.run({})
end_time = time.time()
inference_speed = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(response)) / (end_time - start_time)
print(f"Inference speed 4 GPUs: {inference_speed:.2f} tokens/second")

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/starchat-beta",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 256},
    device = -1
    )
llm_chain = LLMChain(prompt=chat_prompt, llm=llm)

start_time = time.time()
response =llm_chain.run({})
end_time = time.time()
inference_speed = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(response)) / (end_time - start_time)
print(f"Inference speed single CPU: {inference_speed:.2f} tokens/second")


openai_chain = LLMChain(prompt=chat_prompt, llm=openai)
start_time = time.time()
response =openai_chain.run({})
end_time = time.time()
inference_speed = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(response)) / (end_time - start_time)
print(f"Inference speed 4 GPUs: {inference_speed:.2f} tokens/second")
