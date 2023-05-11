import os 
import openai
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.output_parsers import CommaSeparatedListOutputParser

def csv_pandas_agent(keywords):
    openai.apikey = os.getenv('OPENAI_API_KEY')
    url =  os.getenv('GCP_kaggle_exercise')
    output_parser = CommaSeparatedListOutputParser()
    #format_instructions = output_parser.get_format_instructions()
    agent = create_csv_agent(OpenAI(temperature=0),url, verbose=True)
    pd_answer = agent.run("show 10 title that related with {}".format(keywords) )
    pd_ans_format = output_parser.parse(pd_answer)
    return pd_ans_format