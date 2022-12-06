import pandas as pd

def get_data():
    """Returns the data as a pandas dataframe"""
    df = pd.read_csv('/Users/mariusstokkedal/Desktop/DV2599_Maskininlärning/DV2599_ML/Ass2/spambase.csv')
    return df

def row_ranker(row: list) -> list:
    score = 1
    score_dict = {}
    for val in set(row):
        count = row.count(val)
        tot = 0
        for i in range(count):
            tot += i+score
        score += count
        score_dict[val] = tot/count
    return [score_dict[val] for val in row]

def average(lst: list):
    return sum(lst)/len(lst)

def friedman(knn: list, svm: list, DT: list):
    avg_rank = (average(knn) + average(svm) + average(DT)) / 3
    return 10 * ((average(knn) - avg_rank) ** 2 + (average(svm) - avg_rank) ** 2 + (average(DT) - avg_rank) ** 2)


data = get_data()
print(friedman(data))