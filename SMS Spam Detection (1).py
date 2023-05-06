#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('spam.csv',encoding='ISO-8859-1')


# In[3]:


df.sample(5)


# # 1.DATA CLEANING

# In[4]:


df.info()


# In[5]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[6]:


df.sample(5)


# In[7]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[8]:


df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()


# In[10]:


df['target']=encode.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


#missing values
df.isnull().sum()


# In[13]:


#duplicate values
df.duplicated().sum()


# In[14]:


df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# # 2.EDA

# In[16]:


df['target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[18]:


import nltk


# In[19]:


nltk.download('punkt')


# In[20]:


df['num_characters']=df['text'].apply(len)


# In[21]:


df.head()


# In[22]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[23]:


df.head()


# In[24]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[25]:


df.head()


# In[26]:


df[['num_characters','num_words','num_sentences']].describe()


# In[27]:


df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[28]:


df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[29]:


import seaborn as sns


# In[30]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[31]:


sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# # 3.Data Preprocessing

# In[32]:


nltk.download('stopwords')


# In[33]:


from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('reading')


# In[34]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for  i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[35]:


transform_text(df['text'][10])


# In[37]:


df['transformed_text']=df['text'].apply(transform_text)


# In[38]:


df.head()


# In[39]:


from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[40]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[41]:


plt.imshow(spam_wc)


# In[42]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)


# In[43]:


df[df['target']==1]['transformed_text'].tolist()


# In[44]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[45]:


len(spam_corpus)


# In[46]:


from collections import Counter


# In[47]:


Counter(spam_corpus).most_common(30)


# In[48]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

len(ham_corpus)


# In[49]:


Counter(ham_corpus).most_common(30)


# # 4.Model Building

# In[132]:


# Text Vectorization
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[133]:


x=tfidf.fit_transform(df['transformed_text']).toarray()


# In[134]:


#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#x=scaler.fit_transform(x)


# In[135]:


#x=np.hstack((x,df['num_characters'].values.reshape(-1,1)))


# In[136]:


x.shape


# In[137]:


y=df['target'].values


# In[138]:


y


# In[139]:


from sklearn.model_selection import train_test_split


# In[140]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[141]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[142]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[143]:


gnb.fit(X_train,Y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))


# In[144]:


mnb.fit(X_train,Y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(Y_test,y_pred2))
print(confusion_matrix(Y_test,y_pred2))
print(precision_score(Y_test,y_pred2))


# In[145]:


bnb.fit(X_train,Y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(Y_test,y_pred3))
print(confusion_matrix(Y_test,y_pred3))
print(precision_score(Y_test,y_pred3))


# In[146]:


# Chosen TFIDF Followed BY MNB


# In[147]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[148]:


svc= SVC(kernel='sigmoid',gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear',penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)
abc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gdbt=GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb=XGBClassifier(n_estimators=50,random_state=2)


# In[149]:


clfs={
    'SVC':svc,
    'KN':knc,
    'NB':mnb,
    'DT':dtc,
    'LR':lrc,
    'RF':rfc,
    'AdaBoost':abc,
    'BgC':bc,
    'ETC':etc,
    'GBDT':gdbt,
    'xgb':xgb
}


# In[150]:


def train_classifier(clf,X_train,Y_train,X_test,Y_test):
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)
    precision=precision_score(Y_test,y_pred)
    
    return accuracy,precision


# In[151]:


# Model Improvement
#1.Change the max_feature parameter of tfidf


# In[152]:


#Voting Classifier
svc=SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb=MultinomialNB()
etc=ExtraTreesClassifier(n_estimators=50 , random_state=2)

from sklearn.ensemble import VotingClassifier


# In[153]:


voting=VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')


# In[155]:


#voting.fit(X_train,Y_train)


# In[157]:


#y_pred=voting.predict(X_test)
#print("Accuracy ",accuracy_score(Y_test,y_pred))
#print("Precision ",precision_score(Y_test,y_pred))


# In[158]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[159]:


import streamlit as st


# In[160]:


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


# In[161]:


st.title("SMS SPAM CLASSIFIER")

input_sms=st.text_input("Enter the message")


# In[162]:


#1.Preprocessing
transformed_sms=transform_text(input_sms)
#2.Vectorize
vector_input=tfidf.transform([transformed_sms])
#3.Predict
model.fit(X_train, Y_train)
result=model.predict(vector_input)[0]
#4.Display
if(result==1):
    st.header("Spam")
else:
    st.header("Not Spam")


# In[ ]:





# In[ ]:




