import json
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser

def set_exercise(pd_ex, search_ex):
    temp = []
    if pd_ex[0]!="'Agent stopped due to iteration limit or time limit.'":
        temp +=pd_ex
    if search_ex[0]!="'Agent stopped due to iteration limit or time limit.'":
        temp+=search_ex 
    ex_list = (", ").join(temp)
    print(ex_list)
    print(pd_ex, search_ex)
    return ex_list

def set_grade_explanation(grade_num):
    grade = {0:"참가증", 1:'3등급', 2:'2등급', 3:'1등급'}
    _1st = "Cardiorespiratory endurance, strength, muscular endurance, and flexibility are all at or above the 70th percentile, and one of agility, quickness, or coordination is at or above the 70th percentile."
    _2nd = "Cardiorespiratory endurance, strength, muscular endurance, and flexibility are all at or above the 50th percentile, and one of agility, quickness, or coordination is at or above the 50th percentile."
    _3nd = "Cardiorespiratory endurance, strength, muscular endurance, and flexibility are all below the 30th percentile and within an appropriate BMI level."
    _4nd = "Cardiorespiratory endurance, muscular strength, muscular endurance, and flexibility are all below the 30th percentile and do not fall within an appropriate BMI level."
    explanation = {0:_4nd, 1:_3nd, 2:_2nd, 3:_1st}
    data = {"grade_num": grade_num, "grade_name": grade[grade_num], "grade_explanation":explanation[grade_num]}
    return data
class Step(BaseModel):
    exercise_list: set[str] = Field(description="list of exercise in certain exercise routine step")
    time: int = Field(description="time to spend for exercise routine certain step")
class Workout_Routine(BaseModel):
    step_1:Step
    step_2:Step
    step_3:Step

def prompt_agent(pd_ex, search_ex, grade):
    ex_list = set_exercise(pd_ex,search_ex)
    health_condition = set_grade_explanation(grade)["grade_explanation"]
    chat = ChatOpenAI(temperature=0)
    template = """"
    You are a helpful assistant that make workout routine using {ex_list} which is called "exercise candidates". Workout routine is make up for 3 steps. 
    client's health condition is {health_condition}.
    For step 1he, we need to do Warm-up and Stretching and it is important to warm up your body and stretch your muscles to prevent injury. 
    For step 2, we need to do main exercise.
    For step 3, it is important to cool down your body and stretch your muscles to prevent soreness and stiffness. 
    Before making workout routine if korean is in exercise candiates, translate it to English.
    tell me what exercise is in step 1, step 2, step 3 and each step have to contain at least 5 exercise from "exercise candidates".
    tell me how much time should be spend for each step 1, step 2, step 3 and just give number for it.
    output to return should be only one json. do not add any description about output.
    For json format, key for step 1 is "step_1" and key for step 2 is "step_2", key for step 3 is "step_3".
    For json format, key "step_1", "step_2". "step_3" compose of exercise list and time.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{ex_list}"
    parser = PydanticOutputParser(pydantic_object=Workout_Routine)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate(messages = [system_message_prompt, human_message_prompt],
                                    partial_variables={"format_instructions": parser.get_format_instructions()}, 
                                    input_variables=["ex_list","health_condition"],)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    output_kor=chain.run(ex_list=ex_list, health_condition=health_condition)
    parser.parse(output_kor)
    return output_kor
