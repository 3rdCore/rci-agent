from langchain import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

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

self.openai = ChatOpenAI(
    model_name="gpt-4",
    **params,
    model_kwargs=open_ai_params,
    openai_api_key="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",
)
template="You are an autoregressive language model that completes user's sentences. You should not conversate with user."
input_prompt = pt

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

print(openai(messages).content)
print("=-"*20)
print(openai_chain.run({}))


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/starchat-beta",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_new_tokens": 64},
    device = -1
    )
llm_chain = LLMChain(prompt=chat_prompt, llm=llm)
print(llm_chain.run({}))
