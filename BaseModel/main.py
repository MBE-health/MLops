from pydantic import BaseModel

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