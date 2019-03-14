
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
filepath_dict = {'control_flow':   'bank/Bank-Controlflow.txt',
                 'data_flow': 'bank/Bank-Dataflow.txt',
                'temporal':'bank/Bank-Temporal.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['rule', 'label'], sep='\t')
    df['label'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
df.head()


# In[16]:


x=df['rule'].values
print(x)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

labels = df['label']
text = df['rule']

X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.1)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_transformed = tf_transformer.transform(X_test_counts)

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

print(labels.classes_)


# In[ ]:





# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train_transformed,y_train_lables_trf)
score = classifier.score(X_train_transformed,y_train_lables_trf)
classifier.predict_proba(X_test_transformed)

print("acuracy:",score)

print('Predicted probabilities of demo input string are')
probab=classifier.predict_proba(p_tfidf)
print("Control flow probability:",probab[:,0])
print("Data flow probability:",probab[:,1])
print("Temporal flow probability:",probab[:,2])


# In[44]:



if probab[:,0] <= 0.5:
    print('the document compliance is for control flow  is low')
elif probab[:,1]<= 0.5: 
    print('the document compliance is dataflow rule is  low')
elif probab[:,2]<= 0.5:
    print("the document compliance temporal is low")

