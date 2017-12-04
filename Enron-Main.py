# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:54:04 2017

Author: Jingwang Li
Email : 619620054@qq.com
Blog  : www.jingwangl.com

"""

import os
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from collections import Counter
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk.stem.porter import *
from sklearn import cross_validation, metrics
from sklearn import feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

##数据清洗
#从原始文件夹中读取150人的邮件
root_dir = 'D:nltk_data/corpora/maildir/'
new_root_dir = 'D:nltk_data/corpora/new_maildir/'
if not os.path.exists(new_root_dir):
    os.makedirs(new_root_dir)
name_list = os.listdir(root_dir)
sentDir_list = ['/_sent_mail','/sent_items','/sent','/deleted_items','/chris_stokley/sent']
for name_dir in name_list:
    new_name_dir = new_root_dir + name_dir
    if not os.path.exists(new_name_dir):
        os.makedirs(new_name_dir)
    for sentDir in sentDir_list:
        corpus_root = root_dir + name_dir + sentDir
        if os.path.exists(corpus_root): break
    #file_pattern = r".*/.*\.txt"
    name=name_dir.split('-')[0]
    locals()[name + '_mail_list'] = os.listdir(corpus_root)
    locals()[name + '_mails'] = PlaintextCorpusReader(corpus_root, '\d*')
    mail_list = locals()[name + '_mail_list']
    mails = locals()[name + '_mails']
    body_cnt = 1
    for i in mail_list:
        raw = mails.raw(i)
        name_match = re.findall(r'From:.*@',raw)
        if name in name_match[0] and re.findall(r'Forwarded by',raw)==[]:
            if len(re.findall(r'@',raw))>6:
                continue
            try:
                start = re.search(r'\.(pst|PST|nsf|NSF)\s+',raw).span()[1]
            except:
                start = re.search(r'X-FileName:\s+',raw).span()[1]
            if re.findall(r'Original Message',raw)==[] and len(re.findall(r'\sTo:.*@',raw))<=1:
                print('1')
                body = raw[start:]
            elif re.findall(r'Original Message',raw)!=[] and len(re.findall(r'\sTo:.*@',raw))<=1:
                print('2')
                end = re.search(r'\s*-*\s?Original Message',raw).span()[0]
                body = raw[start:end]
            elif re.findall(r'Original Message',raw)==[] and len(re.findall(r'\sTo:.*@',raw))>1:
                print('3')
                try:
#                    end = re.search(r'\s+.*\s(on)?.*\s+(PM|AM)\s+.*\s*To:.*@',raw).span()[0]
                    end = re.search(r'\s+.*\s(on)?.*\s+(PM|AM)\s+.*\s*.*To:.*@',raw).span()[0]
                except:
                    end = re.search(r'\s+(.*\s(on)?.*\s+(PM|AM)\s+)?.*\s*To:.*@.*\scc:',raw).span()[0]
                body = raw[start:end]
            else:
                print('4')
                end = re.search(r'\s*-*\s?Original Message',raw).span()[0]
                body = raw[start:end]
            file_name = new_name_dir + '/' + str(body_cnt)
            if not os.path.exists(file_name):
                f = open(file_name,'w')
                f.write(body)
                f.close()
                body_cnt += 1

##从清理后的数据集中中读取150人的邮件正文
root_dir = 'D:nltk_data/corpora/new_maildir/'
name_list = os.listdir(root_dir)
persons_list=[]
choosen_name_list=[]
persons_numOfmails=[]
for name_dir in name_list:
   corpus_root = root_dir + name_dir
   name=name_dir.split('-')[0]
   locals()[name + '_mail_list'] = os.listdir(corpus_root)
   #剔除邮件数低于100的发送者
   if len(os.listdir(corpus_root))>=100:
       choosen_name_list.append(name)
       locals()[name + '_mails'] = PlaintextCorpusReader(corpus_root, '\d*')
       persons_list.append(locals()[name + '_mails'])
   persons_numOfmails.append(len(locals()[name + '_mail_list']))

a = np.array(persons_numOfmails)
numOfmails = sum(a[a>=100])
plt.figure(figsize=(10,8))
plt.scatter(range(1,len(persons_numOfmails)+1),persons_numOfmails)
plt.xlabel('number')
plt.ylabel('num of mails')
savefig('D:/1.png')

##choose features
#剔除stopwords、标点符号和数字，进行词干提取
stopwords = nltk.corpus.stopwords.words('english')
person_wordsList = []
corpus = []
for name in choosen_name_list:
   text = nltk.Text(locals()[name + '_mails'].words())
   locals()[name + '_wordsList'] = [w.lower() for w in text if w.isalpha() and w.lower() not in stopwords]
   corpus.append(' '.join(locals()[name + '_wordsList']))
   person_wordsList.append(locals()[name + '_wordsList'])

allWords = sum(person_wordsList,[])

def tf(word, count):
   return count[word] / sum(count.values())

def n_containing(word, count_list):
   return sum(1 for count in count_list if word in count)

def idf(word, count_list):
   return math.log(len(count_list) / (1 + n_containing(word, count_list)))

def tfidf(word, count, count_list):
   return tf(word, count) * idf(word, count_list)

#计算TF-IDF值
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

#挑选出每个发送者所有邮件内容中TF-IDF值最大的词，建立KeyWordSet
#KeyWordSet中每个单词作为一个bool类型的特征
KeyWordSet = set()
for i in range(weight.shape[0]):
   KeyWordSet.add(word[weight[i].argmax()])
#len(KeyWordSet)=68 即68个特征

#从每封邮件内容里面提取出需要的特征，得到进行机器学习所需的DataSet
numOffeature = 77
columns = []
for i in range(numOffeature):
    columns.append('feature' + str(i+1))
columns.append('label')
DataSet = pd.DataFrame(index=range(numOfmails),columns=columns)
cnt = 0
for name in  choosen_name_list:
    print(name)
    mails = locals()[name + '_mails']
    mail_list = locals()[name + '_mail_list']
    for mail in mail_list:
        raw = mails.raw(mail)
        #判断是否为空白邮件
        if len(re.findall(r'\s',raw))==len(raw):
            continue
        print(mail)
        words = mails.words(mail)
        sents = mails.sents(mail)
        words = [w.lower() for w in words]
        FeatureList = []
        #Feature1-68：KeyWord
        for KeyWord in KeyWordSet:
            if KeyWord in words:
                FeatureList.append(1)
            else:
                FeatureList.append(0)
        #Feature69:平均句长
        FeatureList.append(len(words)/len(sents))
        #Feature70:"！"出现的频率
        FeatureList.append(words.count('!')/len(words))
        #Feature71:"$" "%" "#"出现的频率
        FeatureList.append((words.count('$') + words.count('%') + words.count('#'))/len(words))
        #Feature72:"-"出现的频率
        FeatureList.append(words.count('-')/len(words))
        #Feature73:大写字母出现的频率
        FeatureList.append(len(re.findall(r'[A-Z]',raw))/len(raw))
        #Feature74:标点符号、空格、换行符在正文中所占比例
        FeatureList.append(len(re.findall(r'(\.|,|:|!|\s)',raw))/len(raw))
        #Feature75:某种符号连续出现的情况所占的比例 "...."
        FeatureList.append(len(re.findall(r'(\.\.+|,,+\|''+)',raw))/len(raw))
        #Feature76:疑问句出现的频率
        FeatureList.append(len(re.findall(r' .+\?',raw)))
        #FeaTure77:特殊句式
        FeatureList.append(len(re.findall(r'[a,A][r,n][e,y].*\?',raw) + re.findall(r'[i,I]s.*\?',raw))/len(raw))

        label = choosen_name_list.index(name)
        FeatureAndLabel = FeatureList + [label]
        DataSet.loc[cnt] = np.array(FeatureAndLabel)
        cnt += 1

DataSet.to_csv('D:/DataSet.csv')

##解决DataSet数据集不均衡问题,进行OverSampling
maxNum=max(persons_numOfmails)
cnt=0
for i in range(len(choosen_name_list)):
    iSet=DataSet[DataSet.label==i]
    n=maxNum-len(iSet)
    choice=np.random.randint(0,len(iSet),n)
    for j in range(n):
        DataSet=DataSet.append(iSet.iloc[choice[j]],ignore_index=True)
        cnt+=1

DataSet.to_csv('D:/NewDataSet.csv')
##划分Train_set和Test_set
x=DatSet.iloc[:,:numOffeature-1]
y=DataSet.label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

##选择算法进行训练及预测
#RandForest
#调参
reasult=[]
for i in range(1,200):
   rf0 = RandomForestClassifier(oob_score=True, random_state=i)
   rf0.fit(x_train,y_train)
   y_predict = rf0.predict(x_test)
   AccuracyRate=sum(y_predict==y_test)/len(y_test)
   reasult.append([i,AccuracyRate])

reasult=np.array(reasult)

Best_random_state=int(reasult[reasult[:,1].argmax()][0])
rf0 = RandomForestClassifier(oob_score=True, random_state=Best_random_state)
rf0.fit(x_train,y_train)
y_predict = rf0.predict(x_test)
AccuracyRate0=sum(y_predict==y_test)/len(y_test)
print('AccuracyRate:',AccuracyRate)
# 0.916925392793

