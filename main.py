# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
from pydantic import BaseModel
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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



def non_rec(group_num, input_parameters):

    import pandas as pd 
    # loading the saved model
    create_CF = joblib.load('./model/rec/non/create_CF.pkl')
    get_CF = joblib.load('./model/rec/non/get_CF.pkl')
    data = joblib.load('./model/rec/user_group/adult_group_{}.pkl'.format(group_num))
    get_freq_item = joblib.load('./model/rec/non/get_freq_item.pkl')
    get_association_rules = joblib.load('./model/rec/non/get_association_rules.pkl')
    get_apriori_result = joblib.load('./model/rec/non/get_apriori_result.pkl')
    get_sorted= joblib.load('./model/rec/non/get_sorted.pkl')
    create_top5_ex = joblib.load('./model/rec/non/create_top5_ex.pkl')
    get_sparse_matrix = joblib.load('./model/rec/non/get_sparse_matrix.pkl')
    get_top5_ex = joblib.load('./model/rec/non/get_top5_ex.pkl')

    similarity_pair=create_CF(data, pd.DataFrame([input_parameters.dict()]))
    pre_ex_list =get_CF(similarity_pair, "준비운동", int(data.shape[0]*(10/100)))
    main_ex_list = get_CF(similarity_pair, "본운동", int(data.shape[0]*(10/100)))
    after_ex_list = get_CF(similarity_pair, "마무리운동", int(data.shape[0]*(10/100)))


    pre_ar_df = get_apriori_result(pre_ex_list,min_support = 0.05, min_confidence=0.8)
    pre_top5 = get_top5_ex(pre_ar_df)

    main_ar_df = get_apriori_result(main_ex_list,min_support = 0.05, min_confidence=0.8)
    main_top5 = get_top5_ex(main_ar_df)
    
    after_ar_df = get_apriori_result(after_ex_list,min_support = 0.05, min_confidence=0.8)
    after_top5 = get_top5_ex(after_ar_df)


def gred_pred(input_parameters : clf_input):
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
     

@app.post('/non_rec')
def non_rec(input_parameters : clf_input):
    group_num = gred_pred(input_parameters)
    rec = non_rec(group_num, input_parameters)
    return group_num



@app.get('/')
def home():
	return {"message": "Welcome Home!"}