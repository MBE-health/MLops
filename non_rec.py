import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def create_CF(df,input_parameters):
  category_col = ["성별구분코드_F", "성별구분코드_M"]
  body_col = ["신장(cm)","체중(kg)","체지방율(%)"]
  # 신장, 체중, 신체 구성, 
  health_col = ["측정연령수","앉아윗몸앞으로굽히기(cm)","BMI(kg/㎡)","교차윗몸일으키기(회)","왕복오래달리기(회)","10M 4회 왕복달리기(초)","제자리 멀리뛰기(cm)","상대악력(%)"]
  col = category_col+body_col + health_col
  new_user_df =  pd.DataFrame([input_parameters.dict()])    
  CF = pd.DataFrame(cosine_similarity(df[col],new_user_df),columns=["similarity"])
  #CF.columns = total_user.index
  return CF

def get_CF(CF,data,  ex_type, top_n):
    idx = CF.sort_values(by = "similarity",ascending = False)[1:top_n].index.values
    ex_list = []
    for i in idx:
        other_viewed_list = data[data.index==i][ex_type].values
        for j in other_viewed_list:
            if j not in ex_list:
              ex_list.append(j.split(","))
    
    #print('====={} 운동 추천 목록 ====='.format(ex_type))
    #for i in ex_list:
      #print(i)
    
    return ex_list

def get_sparse_matrix(df):
  te = TransactionEncoder()
  te_ary = te.fit(df).transform(df)
  sparse_matrix = pd.DataFrame(te_ary, columns=te.columns_) #위에서 나온걸 보기 좋게 데이터프레임으로 변경
  return sparse_matrix

def get_freq_item(sparse_matrix, min_support): # 0.01
  #print("시작")
  frequent_itemsets = apriori(sparse_matrix, min_support=min_support, use_colnames=True)
  #print("완료")
  return frequent_itemsets


def get_association_rules(frequent_itemsets, min_threshold): #0.8
  association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
  return association_rules_df


def get_apriori_result(result,min_support, min_confidence):
  sparse_matrix = get_sparse_matrix(result)
  frequent_itemsets = get_freq_item(sparse_matrix, min_support)
  association_rules_df = get_association_rules(frequent_itemsets,min_confidence)
  return association_rules_df


# consequents 1개인 거 -> 지지도, 신뢰도, lift 내림차순
def get_sorted(ar_df):
  ar_df_sorted = ar_df[ar_df["consequents"].apply(lambda x : len(x) == 1)].sort_values(by=["support","confidence","lift"], ascending=False)
  ar_df_sorted["consequents"] = ar_df_sorted["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode").apply(lambda x: x.split(','))
  ar_df_sorted["antecedents"] = ar_df_sorted["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode").apply(lambda x: x.split(','))
  return ar_df_sorted

def create_top5_ex(ar_df_sorted):
  total = []
  for idx in ar_df_sorted.index:
    temp = ar_df_sorted.loc[idx,'consequents']+ar_df_sorted.loc[idx,'antecedents'] 
    for ex  in temp:
      if ex not in total:
        total.append(ex)
  length = 5 if len(total) else len(total) 
  return total[:length]


def get_top5_ex(ar_df):
  ar_df_sorted = get_sorted(ar_df)
  ex_rec_list = create_top5_ex(ar_df_sorted)
  return ex_rec_list


def get_5_ex(ex_list):
  
  if len(ex_list)!=5:
    while len(ex_list)!=5:
      idx = random.randint(0, len(ex_list))
      ex_list.append(ex_list[idx])
  return ex_list
