# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
from .BaseModel import clf_input 

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






@app.post('/grade_pred')
def grade_pred(input_parameters : clf_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    # loading the saved model
    grade_clf_model = joblib.load('./model/grade_clf/clf_LGBM.pkl')

    
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
    
    prediction = grade_clf_model.predict([input_list])
    return prediction[0].item()


@app.get('/')
def home():
	return {"message": "Welcome Home!"}