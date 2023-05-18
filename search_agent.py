import os
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

def set_tools():
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaAPIWrapper()

    wikipedia_tool = Tool(
        name='wikipedia',
        func= wikipedia.run,
        description="Useful for when you need to look up a topic, heath or exercise, fitness on wikipedia"
    )

    duckduckgo_tool = Tool(
        name='DuckDuckGo Search',
        func= search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
    )
    tools = [ duckduckgo_tool, wikipedia_tool]
    return tools


def set_memory():
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=4,
        return_messages=True
    )
    return memory

def set_agent(tools, llm, memory):
    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description", 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=3,
        memory=memory
    )
    return zero_shot_agent


def set_prompt(format_instructions):
    prompt = PromptTemplate(
        template="List 10 sports in noun format which is related with {subject} f.\n{format_instructions}",
        #template="List five exercise similar with {subject} f.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )

    return prompt

def set_format():
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    format = {"output_parser":output_parser, "format_instructions":format_instructions}
    return format

def search_tools_agent(keyword):
    #openai.apikey = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(temperature=0.5)
    tools = set_tools()
    memory = set_memory()
    agent = set_agent(tools, llm, memory)
    format = set_format()
    prompt = set_prompt(format["format_instructions"])

    _input = prompt.format(subject=keyword) # {키워드}한 운동 추천
    search_ans=agent(_input)
    search_ans_format = format["output_parser"].parse(search_ans["output"])
    return search_ans_format
     