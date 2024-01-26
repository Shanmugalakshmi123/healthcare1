from imblearn.over_sampling import SMOTE
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
def rating_preprocessing(df3):
    for i,row in df3.iterrows():
        if row['Rating']==0:
            rating=-1
        elif row['Rating']==1:
            rating=-1
        elif row['Rating']==2:
            rating=-1
        elif row['Rating']==3:
            rating=0
        elif row['Rating']==4:
            rating=1
        elif row['Rating']==5:
            rating=1
        df3.at[i,'Rating1']=rating
    return df3
def review_preprocessing(df3):
    for i,row in df3.iterrows():
        row['Review_Text']=row['Review_Text'].lower()
        row['Review_Text']=row['Review_Text'].replace('\n',' ')
        row['Review_Text']=row['Review_Text'].replace('  ',' ')
        word_token=nltk.word_tokenize(row['Review_Text'])
        e=stopwords.words('english')
        n=[word for word in word_token if word not in e]
        df3.at[i,'Review_Text']=n
    oht=df3['Review_Text'].str.join('|').str.get_dummies('|')
    result1=pd.concat([df3,oht],axis=1) 
    return result1

def predict_test(result1,test,oht):
    x=result1.iloc[:,5:40]
    y=result1.iloc[:,1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    clf=DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    print(cm)
    print(acc)
    treee=tree.export_text(clf)
    print(treee)
    feat=clf.feature_importances_
    print(feat)
    test=test.lower()
    test=nltk.word_tokenize(test)
    e=stopwords.words('english')
    n=[word for word in test if word not in e]
    x1=[]
    x2=[]
    m=oht.columns.tolist()
    g=m[5:40]
    for i in g:
        if i in n:
            x1.append(1)
        else:
            x1.append(0)
    x2.append(x1)
    pred=clf.predict(x2)
    return pred

def predict_knn(result1,test):
    sm=SMOTE(random_state=0,k_neighbors=5)
    x=result1.iloc[:,5:40]
    #x=result1.iloc[:,[9,15,38,27,8,20,21,16,35,36,25]]
    y=result1.iloc[:,1]
    x_res,y_res=sm.fit_resample(x,y)
    x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.3,random_state=42)
    clf=KNeighborsClassifier()
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    print(cm)
    print(acc)
    test=test.lower()
    test=nltk.word_tokenize(test)
    e=stopwords.words('english')
    n=[word for word in test if word not in e]
    x1=[]
    x2=[]
    g=[]
    m=result1.columns.tolist()
    # g.append(m[9])
    # g.append(m[15])
    # g.append(m[38])
    # g.append(m[27])
    # g.append(m[8])
    # g.append(m[20])
    # g.append(m[21])
    # g.append(m[16])
    # g.append(m[35])
    # g.append(m[36])
    # g.append(m[25])
    g=m[5:40]
    for i in g:
        if i in n:
            x1.append(1)
        else:
            x1.append(0)
    x2.append(x1)
    #pred=x2
    #st.write(x2)
    pred=clf.predict(x2)
    return pred



df=pd.read_csv("healthcare_reviews.csv")
df2=df[df.Review_Text.notnull()]
df3=df2[df2.Rating.notnull()]
df3=rating_preprocessing(df3)
result1=review_preprocessing(df3)
test=st.text_input("Enter Text")
#pred=predict_test(result1,test,result1)
if st.button("Predict"):
    pred=predict_test(result1,test,result1)
    #pred=predict_knn(result1,test)
    st.write(pred)
    if pred==4 or pred==5:
        st.write("Positive")
    elif pred==0 or pred==1 or pred==2:
        st.write("Negative")
    elif pred==3:
        st.write("Neutral")
