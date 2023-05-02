import os
import openai
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_ex_by_factor(factor):

    openai.apikey = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(temperature=1,model_name="text-davinci-003")
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
    )

    duckduckgo_tool = Tool(
        name='DuckDuckGo Search',
        func= search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
    )
    tools =[ duckduckgo_tool, wikipedia_tool]
    # conversational agent memory
    memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3,
        return_messages=True
    )
    #set Agent
    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description", 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=3,
        memory=memory
    )
    # set Chat
    chat = ChatOpenAI(temperature=0)
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    
    # set Prompt and output parser
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List five fitness sport in noun format which is good for {subject} f.\n{format_instructions}",
        #template="List five exercise similar with {subject} f.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )

    _input = prompt.format(subject=factor) # {키워드}한 운동 추천
    output_text=zero_shot_agent(_input)
    output_kor=chain.run(input_language="English", output_language="Korean", text=output_text["output"])
    answer = output_parser.parse(output_kor)[:5]
    return answer
