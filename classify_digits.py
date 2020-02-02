#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class KNNClassifier:
    train_data = pd.DataFrame()
    k = None
    
    def train_validation_split(self,data_frm,validation_data_size):
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))

        indices=data_frm.index.tolist()

        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]

        train_datafrm=data_frm.drop(valid_indices)

        return train_datafrm, valid_datafrm
    
    def createDistanceLabelEuclidean(self, test_sample):
        results_dist_label=[]
        for li in self.train_data:
            test_list=test_sample[0:]
            train_list=li[1:]
            dist=np.linalg.norm(test_list-train_list)
            results_dist_label.append([dist,li[0]])

        results_dist_label.sort()

        return results_dist_label
    
    def createDistanceLabelManhattan(self, test_sample):
        results_dist_label=[]
        for li in self.train_data:
            test_list=test_sample[0:]
            train_list=li[1:]
            dist=np.sum(np.absolute(test_list - train_list))
            results_dist_label.append([dist,li[0]])

        results_dist_label.sort()

        return results_dist_label
    
    def getPredictedLabelValue(self,results_dist_label):
        label_count={}
        for i in range(self.k):
            val = results_dist_label[i][1]
            if val in label_count:
                label_count[val]+=1
            else:
                label_count[val]=1

    #     for ky,vl in label_count.items():
    #         print(ky,":",vl)

        return max(label_count,key=label_count.get)
    
    
    def getPredictedLabels(self, validation_data):
        predicted_list=[]

        for test_sample in validation_data:
            results_dist_label = self.createDistanceLabelEuclidean(test_sample)
            predicted_label = self.getPredictedLabelValue(results_dist_label)
            predicted_list.append(predicted_label)
        return predicted_list
    
    def check_validation(self,train_data_frm, validation_data_size):
        random.seed(0)
        train_data_frm , validation_data_frm = self.train_validation_split(train_data_frm, validation_data_size)
        self.train_data = train_data_frm.values
        
        validation_data_labels = validation_data_frm.iloc[:,0].to_frame().values.tolist()
        validation_data_frm = validation_data_frm.drop([validation_data_frm.columns[0]],  axis='columns')
        validation_data = validation_data_frm.values
        predicted_labels = self.getPredictedLabels(validation_data)
        print(accuracy_score(validation_data_labels, predicted_labels))
        
   
    def train(self,train_data_path):
        train_data_frm = pd.read_csv(train_data_path)
        self.k=3
#         self.check_validation(train_data_frm, validation_data_size = 100)
        self.train_data = train_data_frm.values
    
    def predict(self,test_path):
        test_data_frm = pd.read_csv(test_path, header=None)
        predicted_labels = self.getPredictedLabels(test_data_frm.values)
        return predicted_labels
               
        


# In[25]:


# knn_classifier = KNNClassifier()
# knn_classifier.train('/home/jyoti/Documents/SMAI/assign1/q1/train.csv')
# predictions = knn_classifier.predict('/home/jyoti/Documents/SMAI/assign1/q1/test.csv')
# test_labels = list()
# with open('/home/jyoti/Documents/SMAI/assign1/q1/test_labels.csv') as f:
#   for line in f:
#     test_labels.append(int(line))
# print (accuracy_score(test_labels, predictions))

