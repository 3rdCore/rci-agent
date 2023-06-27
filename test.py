from langchain import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, SimpleChatModel
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
    openai_api_key="",
)

messages = [
    SystemMessage(
        content="You are an autoregressive language model that completes user's sentences. You should not conversate with user."
    ),
    HumanMessage(content="Hello, how are you?"),
]

response = openai(messages).content

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/starchat-beta",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
    #device = 0
)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
openai_chain = LLMChain(prompt=prompt, llm=openai)

question = "Can you wrint a python program to print 'hello word' ?"

print(llm_chain.run(question))
print(openai_chain.run(question))