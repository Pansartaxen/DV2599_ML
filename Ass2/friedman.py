from scipy import stats

def row_ranker(row: list) -> list:
    """
    Takes a list as input and returns a list with
    the ranks of the values where 1 is lowest
    """
    # score = 1
    # score_dict = {}
    # for val in set(row):
    #     count = row.count(val)
    #     tot = 0
    #     for i in range(count):
    #         tot += i+score
    #     score += count
    #     score_dict[val] = int(tot/count)
    # return([score_dict[val] for val in row])
    row_cpy = row.copy()
    row_cpy = sorted(row_cpy)
    # row_cpy = list(reversed(row_cpy))
    return_list = [0 ,0 ,0]
    # print("ORG",[score_dict[val] for val in row])
    # print("NEW",[row_cpy.index(val)+1 for val in row])
    if row == [row[0], row[0], row[0]]:
        return [2, 2, 2]

    for index in range(len(row_cpy)):
        if row[index] == row[index-1]:
            rank = int((row_cpy.index(row[index])+1 + row_cpy.index(row[index-1])+1) / 2)
            return_list[index], return_list[index-1] = rank, rank
        else:
            return_list[index] = row_cpy.index(row[index])+1
    print(return_list)
    return return_list
    # return [row_cpy.index(val)+1 for val in row]

def average(lst: list) -> float:
    """Takes in a list and returns the average value"""
    return sum(lst)/len(lst)

def friedman(knn: list, svm: list, DT: list) -> float:
    """Returns the friedman test statistic"""

    for i in range(len(knn)):
        ranked = row_ranker([knn[i], svm[i], DT[i]])
        knn[i] = ranked[0]
        svm[i] = ranked[1]
        DT[i] = ranked[2]

    avg_rank = (average(knn) + average(svm) + average(DT)) / 3
    return 10 * ((average(knn) - avg_rank) ** 2 + (average(svm) - avg_rank) ** 2 + (average(DT) - avg_rank) ** 2)

if __name__ == "__main__":
    knn = [0.6809, 0.7017, 0.7012, 0.6913, 0.6333, 0.6415, 0.7216, 0.7214, 0.6578, 0.7865]
    svm = [0.7164, 0.8883, 0.8410, 0.6825, 0.7599, 0.8479, 0.7012, 0.4959, 0.9279, 0.7455]
    DT = [0.7524, 0.8964, 0.6803, 0.9102, 0.7758, 0.8154, 0.6224, 0.7585, 0.938, 0.7524]
    print(friedman(knn, svm, DT))
    print(stats.friedmanchisquare(knn, svm, DT))
