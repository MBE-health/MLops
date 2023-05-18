# -*- coding: utf-8 -*-
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, File, UploadFile
from typing import Optional
import joblib
import json
from pydantic import BaseModel
from non_rec import *
from search_agent import *
from csv_agent import *
from food_clf import *
from prompt_agent import prompt_agent
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os 
import openai


openai.apikey = os.getenv('OPENAI_API_KEY')
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class clf_input(BaseModel):
    
    측정연령수 : int
    신장 : float
    체중 : float
    체지방율: float
    앉아윗몸앞으로굽히기 : float
    BMI : float
    교차윗몸일으키기 :  float
    왕복오래달리기 : int
    왕복달리기_10M_4회:float
    제자리_멀리뛰기:float
    상대악력:float
    성별구분코드_F:int
    성별구분코드_M:int

def non_ex(group_num, new_user_df):

    #data = joblib.load('./user_group/adult_group_{}.pkl'.format(group_num))
    url =  os.getenv('adult_group_{}'.format(group_num))
    data=pd.read_csv(url)
    similarity_pair=create_CF(data, new_user_df)
    pre_ex_list =get_CF(similarity_pair, data, "준비운동", int(data.shape[0]*(10/100)))
    main_ex_list = get_CF(similarity_pair, data,  "본운동", int(data.shape[0]*(10/100)))
    after_ex_list = get_CF(similarity_pair,data, "마무리운동", int(data.shape[0]*(10/100)))


    pre_ar_df = get_apriori_result(pre_ex_list,min_support = 0.05, min_confidence=0.8)
    pre_top5 = get_top5_ex(pre_ar_df)

    main_ar_df = get_apriori_result(main_ex_list,min_support = 0.05, min_confidence=0.8)
    main_top5 = get_top5_ex(main_ar_df)
    
    after_ar_df = get_apriori_result(after_ex_list,min_support = 0.05, min_confidence=0.8)
    after_top5 = get_top5_ex(after_ar_df)

    ex = {"pre_ex":pre_top5, "main_ex":main_top5, "after_ex":after_top5}
    print(ex)
    return ex

def parse_grade_input(input_parameters : clf_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    
    age = input_dictionary['측정연령수']
    height = input_dictionary['신장']
    weight = input_dictionary['체중' ]
    fat = input_dictionary['체지방율']
    stretch = input_dictionary['앉아윗몸앞으로굽히기']
    bmi = input_dictionary['BMI']
    crossUp = input_dictionary['교차윗몸일으키기' ]
    runLong = input_dictionary['왕복오래달리기']
    runRepeat = input_dictionary[ '왕복달리기_10M_4회']
    jump = input_dictionary['제자리_멀리뛰기']
    armStrength = input_dictionary['상대악력']
    sex_F = input_dictionary[ '성별구분코드_F']
    sex_M = input_dictionary[ '성별구분코드_M']


    input_list = [age, height, weight, fat, stretch, bmi, crossUp, runLong, runRepeat, jump, armStrength, sex_F, sex_M ]

    return input_list

def get_clf(input_list):
    # loading the saved model
    grade_clf_model = joblib.load('./model/grade_clf/clf_LGBM.pkl')
    prediction = grade_clf_model.predict([input_list])
    return prediction[0].item()
     

@app.post('/non_rec')
def non_rec(health_params : clf_input):
    group_num = get_clf(parse_grade_input(health_params))  
    rec = non_ex(group_num, health_params)
    return {"group_num":group_num, "ex":rec}


@app.get("/search_rec") # 검색 툴 활용한
def factor_rec(keyword:str):
    search_rec = search_tools_agent(keyword)
    return {"factor": keyword,"exercise":search_rec }

@app.get("/csv_rec") # csv 데이터 기반
def csv_rec(keywords:str):
    csv_rec = csv_pandas_agent(keywords)
    return {"factor":keywords, "exercise":csv_rec}

@app.post("/total_rec")
def get_total_rec(grade:int, keywords_ex: list[str],condition_ex: list[str] ):
    #grade = get_clf(parse_grade_input(health_params))
    #csv_ex = csv_pandas_agent(csv_keywords)
    total_rec = prompt_agent(condition_ex, keywords_ex, grade)
    return total_rec

'''
@app.post("/predict")
async def pred(file:bytes = File(...)):
  image = read_image(file)
  image = preprocessing(image)
  prediction = pred(image)
  print(prediction)
  return prediction
'''
@app.get('/')
def home():
	return {"message": "Welcome Home!"}