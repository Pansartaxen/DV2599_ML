#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:43:40 2020

@author: saimanojk1
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.preprocessing import KBinsDiscretizer

#Data Preprocessing

#Data Extraction from spambase.data into spambase_data.csv file

with open ('spambase_data.csv', 'w+') as file:
    for i in range(1, 58):
        file.write(str(str(i)+',').rstrip('\n'))
    file.write(str(str(58)).rstrip('\n'))
    file.write('\n')
    with open('spambase.data', 'r') as datafile:
        for line in datafile:
            file.write(line)

#Reading spambase_data.csv into pandas dataframe   
         
df = pd.read_csv('spambase_data.csv', delimiter = ',')

#Dropping columns with null values.

df = df.dropna(axis=1) #There were no null values

# Columns 1 to 57 have continuous real data. 

#Checking for null values.

df2 = df.any()                 #Obtained false for every column. Hence, there are no null values.
print(df2)        

# Converting continuous values to discrete values. Performing a uniform discretization transform of the dataset
# uniform discretization transform will preserve the probability distribution of each input 
#variable but will make it discrete with the specified number of ordinal groups or labels.

print(df.shape)
df.hist()
pyplot.show()

trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
df.iloc[:,:-3] = trans.fit_transform(df.iloc[:,:-3])
print(df)




spam_positives = df.iloc[:1812]
spam_negatives = df.iloc[1813:]

'''
 function for  coverting all values in a 2D list to integer
 Input: 2D list
 Output: 2D list with  int values
 '''
def to_int(list_2d):
    for i in range(len(list_2d)):
        for j in range(len(list_2d[i])):
            list_2d[i][j] = (int)(list_2d[i][j])
    return list_2d

'''
 function for  coverting all values in a 1D list to integer
 Input: 1D list
 Output: 1D list with  int values
 '''
def to_int1(list_1d):
    for i in range(len(list_1d)):
            list_1d[i] = (int)(list_1d[i])
    return list_1d

'''
 function for giving column values in a 2d list
 Input: 2D list, column number
 Output: 1D lst of column values
 '''
def column(matrix, i):
    return [row[i] for row in matrix]


'''
 function for calculating the total number of possiblities for a spam dataset
 Input: data of shape n * 57 * 1
 Output: A 2D list of unique values for each feature in the data and and Integer size of total possible instances
 the features observed
 '''
def possible_space_size(data):
    
    unique_values = []
    
    for j in range(len(data[0])-2):
        unique_values.append([0,1,2,3,4,5,6,7,8,9])
        
    for j in range(len(data[0])-2, len(data[0])):
        unique_values.append(list(set(column(data, j))))

    hyp_space_size = 1
    
    for i in range(len(unique_values)-2):
        hyp_space_size = hyp_space_size * 10 #Since Continuous data is divided into 10 descrete values through binning
    
    for i in range(len(unique_values)-2, len(unique_values)):
        hyp_space_size = hyp_space_size * len(unique_values[i])
        print("\nUnique values ",i+1)
        print(len(unique_values[i]))
    return unique_values, hyp_space_size

'''
 function for calculating size of a given hypothesis space
 Input: data of shape  57 * n
 Output: Integer size of hypothesis space
 '''
def hyp_space_size(data, unique_vals):
    
    hyp_space_size = 1

    for i in range(len(data)-2):
        if data[i] == []:
           hyp_space_size = hyp_space_size * 10
        else:
            hyp_space_size = hyp_space_size * (len(data[i]))

    for i in range(len(data)-2, len(data)):
            if data[i] == []:
               hyp_space_size = hyp_space_size * (len(unique_vals[i]))
            else:
                hyp_space_size = hyp_space_size * (len(data[i]))
    return hyp_space_size

#Since training is only done on positive cases in Concept learning
train_data, test_data = train_test_split(spam_positives, test_size=0.2, random_state = 0)

#Negatives are added to test data
test_data = test_data.append(spam_negatives)

x_test = test_data.iloc[:,:-1].values.tolist()
x_test = to_int(x_test)

y_test = test_data.iloc[:,-1].values.tolist()
y_test  = to_int1(y_test)

#This contains total_data values
total_data = df.iloc[:,:-1].values.tolist()
total_data_y = df.iloc[:,-1].values.tolist()

train_data = train_data.iloc[:,:-1].values.tolist()
train_data = to_int(train_data)

#This contains all the positives
positivesX_test = spam_positives.iloc[:,:-1].values.tolist()
positivesX_test = to_int(positivesX_test)

positivesY_test = spam_positives.iloc[:,-1].values.tolist()
positivesY_test = to_int1(positivesY_test)





#Size of most general hypothesis space
unique_values, hyp_size = possible_space_size(total_data)

print("Size of most general hypothesis space = ", hyp_size)

#Applying Algorithm 4.1
'''
 function for applying algorithm 4.1: LGG_Set()
 Input: data of shape  n * 57 * 1, int m value for implementing algorithm 4.2 or 4.3
 Output: hypothesis space of shape 57 * n
 '''
def LGG_Set(data, m):
    h = []
    #Initiating hypothesis with first instance from Data
    for i in range(len(data[0])):
        h.append([])
        h[i].append(data[0][i])
    for i in range(len(data)):
        x = data[i]
        if m == 2:
            h = LGG_Conj(x, h)
        elif m == 3:
            h = LGG_Conj_ID(x, h)
        
    return h

#Algorithm 4.2: Conjunction of all literals common
'''
 function for applying algorithm 4.2: LGG_Conj() - Conjunction of common literals
 Input: data of shape  57 * 1, 57 * n (feature values of an instance and hypothesis)
 Output: hypothesis space of shape 57 * n
 '''
def LGG_Conj(x, y):
    for i in range(len(x)):
        if y[i] != []:
            if x[i] not in set(y[i]):
                y[i] = []  #Conjunction of common literals
    return y

#Algorithm 4.3 : Internal Disjunction of conjunctions
'''
 function for applying algorithm 4.3: LGG_Conj_ID() - Internal disjunction of conjunctions
 Input: data of shape  57 * 1, 57 * n (feature values of an instance and hypothesis)
 Output: hypothesis space of shape 57 * n
 '''
def LGG_Conj_ID(x, y):
    for i in range(len(x)):
        if x[i] not in set(y[i]):
            y[i].append(x[i])  #Internal disjunction
    return y




'''
 function for testing the hypothesis space (concept learned) on test data
 Input: data of shape n * 57 * 1, 57 * m (several instance values and hypothesis space)
 Output: results of size n 
 '''
def test(data, h):
    result = []
    for i in range(len(data)):
        result.append(1)
        for j in range(len(data[i])):
            if h[j] != []:                  #The hypothesis space has empty lists for Algorithm 4.2 - It says the feature is general
                if data[i][j] not in h[j]:
                    result[-1] = 0
    return result

'''
 function for finding accuracy of the results on test data
 Input: data of shape 57 * 1, 57 * 1 (Ground Truth and Result)
 Output: Accuracy value 
 '''
def find_accuracy(y, result):
    correct_pred = 0
    incorrect_pred = 0
    for i in range(len(result)):
        if result[i] == y[i]:
            correct_pred += 1
        else:
            incorrect_pred += 1
    accuracy = correct_pred /(correct_pred + incorrect_pred)
    return accuracy


#Training Alg 4.1 & 4.2 on 80% positives
hypothesis_space1 = LGG_Set(train_data,2)
print(hypothesis_space1)
hyp_size2 = hyp_space_size(hypothesis_space1, unique_values)
print(hyp_size2)

#Testing on test data
result = test(x_test, hypothesis_space1)
accuracy = find_accuracy(y_test, result)
print(accuracy)

#Testing on all positives
result = test(positivesX_test, hypothesis_space1)
accuracy = find_accuracy(positivesY_test, result)
print(accuracy)

#Testing on all data
result = test(total_data, hypothesis_space1)
accuracy = find_accuracy(total_data_y, result)
print(accuracy)




#Training Alg 4.1 & 4.3 on 80% positives
hypothesis_space2 = LGG_Set(train_data,3)
print(hypothesis_space2)
hyp_size2 = hyp_space_size(hypothesis_space2, unique_values)
print(hyp_size2)

#Testing on test data
result = test(x_test, hypothesis_space2)
accuracy = find_accuracy(y_test, result)
print(accuracy)

#Testing on all positives
result = test(positivesX_test, hypothesis_space2)
accuracy = find_accuracy(positivesY_test, result)
print(accuracy)

#Testing on all data
result = test(total_data, hypothesis_space2)
accuracy = find_accuracy(total_data_y, result)
print(accuracy)




#Training Alg 4.1 & 4.2 on 100% positives
hypothesis_space3 = LGG_Set(positivesX_test,2)
print(hypothesis_space3)
hyp_size2 = hyp_space_size(hypothesis_space3, unique_values)
print(hyp_size2)

#Testing on test data
result = test(x_test, hypothesis_space3)
accuracy = find_accuracy(y_test, result)
print(accuracy)

#Testing on all positives
result = test(positivesX_test, hypothesis_space3)
accuracy = find_accuracy(positivesY_test, result)
print(accuracy)

#Testing on all data
result = test(total_data, hypothesis_space3)
accuracy = find_accuracy(total_data_y, result)
print(accuracy)






#Training Alg 4.1 & 4.3 on 100% positives
hypothesis_space4 = LGG_Set(positivesX_test,3)
print(hypothesis_space2)
hyp_size2 = hyp_space_size(hypothesis_space4, unique_values)
print(hyp_size2)

#Testing on test data
result = test(x_test, hypothesis_space4)
accuracy = find_accuracy(y_test, result)
print(accuracy)

#Testing on all positives
result = test(positivesX_test, hypothesis_space4)
accuracy = find_accuracy(positivesY_test, result)
print(accuracy)

#Testing on all data
result = test(total_data, hypothesis_space4)
accuracy = find_accuracy(total_data_y, result)
print(accuracy)



