import pandas as pd
from sklearn.model_selection import train_test_split

def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('spambase.csv')
    return df

def lgg_conj(H : pd.DataFrame, x : pd.DataFrame):
    """Returns the least general generalization of H and x"""
    y = H.copy()
    for col in range(H.shape[0]):
        if H.iloc[col] == x.iloc[col] and H.iloc[col]: # If the values are the same and not 0
            y.iloc[col] = 1
        else:
            y.iloc[col] = 0
    return y

def lgg_set(data : pd.DataFrame):
    """Returns the least general generalization of the data"""
    H = data.iloc[0] # Initialize H with the first row
    for row in range(data.shape[0]):
        x = data.iloc[row] 
        H = lgg_conj(H, x)
    return H

def continuous_to_discrete(data : pd.DataFrame):
    """Returns the data as a discrete dataframe"""
    retDF = data.copy()
    for col in data.columns:
        retDF[col] = retDF[col].apply(lambda x: 1 if float(x) > 0.01 else 0) # If the value is above 0.01, set it to 1, else 0
    return retDF

def test_performance(data: pd.DataFrame, H: pd.Series):
    """Returns the accuracy of the hypothesis"""
    spam_as_spam = 0
    spam_as_ham = 0
    ham_as_ham = 0
    ham_as_spam = 0

    correct = 0

    H_values = []
    for keys in H.keys():
        if H[keys] == True:
            H_values.append(keys)

    print(f"H-values: {H_values[:-4]}")
    for row in range(data.shape[0]):
        spam = True
        for col in range(data.shape[1]-1): # -1 because the last column is the class
            if H.iloc[col]: 
                if not data.iloc[row, col]: # If the value is 0 and the h-value for the same column is 1
                    spam = False 
        if spam: # If the model says it's spam
            if data.iloc[row, -1]: # If it's actually spam
                spam_as_spam += 1
                correct += 1
            else: # If it's actually ham
                ham_as_spam += 1
        else: # If the model says it's ham
            if data.iloc[row, -1]: # If it's actually spam
                spam_as_ham += 1
            else: # If it's actually ham
                ham_as_ham += 1
                correct += 1

    print("good Spam as spam: ", spam_as_spam)
    print("bad Spam as ham: ", spam_as_ham)
    print("good Ham as ham: ", ham_as_ham)
    print("bad Ham as spam: ", ham_as_spam)
    print(f"Accuracy: {100*round(correct/data.shape[0], 2)} %")
    return correct/data.shape[0]

if __name__ == '__main__':
    print('Starting up!')
    print('Code by Marius Stokkedal and Sebastian Bengtsson!')
    df = get_data()

    spam = df[df['spam'] == 1].copy()
    ham = df[df['spam'] == 0].copy()

    initiated = True

    while initiated or len(h[h==1]) < 5: # make sure h is not empty 
        initiated = False
        train_data, split_data = train_test_split(spam, test_size=0.997)

        train_data = continuous_to_discrete(train_data)
        split_data = continuous_to_discrete(split_data) 

        h = lgg_set(train_data)

    test_performance(pd.concat([ham, split_data]), h) # Test the performance on the split spam data and the ham data
    print("Done!")
