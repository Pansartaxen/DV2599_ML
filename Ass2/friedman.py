import math
def row_ranker(row: list) -> list:
    """
    Takes a list as input and returns a list with the ranks
    of the values where 1 is asigned to the lowest value
    """
    row_cpy = row.copy()
    row_cpy = sorted(row_cpy)
    return_list = [0 ,0 ,0]

    if row == [row[0], row[0], row[0]]:
        return [2, 2, 2]

    for index in range(len(row_cpy)):
        if row[index] == row[index-1]:
            rank = int((row_cpy.index(row[index])+1 + row_cpy.index(row[index-1])+1) / 2)
            return_list[index], return_list[index-1] = rank, rank
        else:
            return_list[index] = row_cpy.index(row[index])+1
    return return_list

def average(lst: list) -> float:
    """Takes in a list and returns the average value"""
    return sum(lst)/len(lst)

def squared_differences(knn: list, svm: list, DT: list, avg_rank: float) -> float:
    sum = 0
    for i in range(len(knn)):
        sum += (knn[i] - avg_rank) ** 2 + (svm[i] - avg_rank) ** 2 + (DT[i] - avg_rank) ** 2
    return sum/20

def friedman(knn: list, svm: list, DT: list) -> float:
    """Returns the friedman test statistic"""

    for i in range(len(knn)):
        ranked = row_ranker([knn[i], svm[i], DT[i]])
        knn[i] = ranked[0]
        svm[i] = ranked[1]
        DT[i] = ranked[2]

    avg_rank = (average(knn) + average(svm) + average(DT)) / 3
    sqr_diff = squared_differences(knn, svm, DT, avg_rank)
    ret = 10 * ((average(knn) - avg_rank) ** 2 + (average(svm) - avg_rank) ** 2 + (average(DT) - avg_rank) ** 2)
    print(sqr_diff)
    return ret / sqr_diff

def critical_difference(alg_1: list, alg_2: list, alg_3: list) -> list:
    """Returns the critical difference"""
    for i in range(len(alg_1)):
        ranked = row_ranker([alg_1[i], alg_2[i], alg_3[i]])
        alg_1[i] = ranked[0]
        alg_2[i] = ranked[1]
        alg_3[i] = ranked[2]

    avg_alg_1 = sum(alg_1) / len(alg_1)
    avg_alg_2 = sum(alg_2) / len(alg_2)
    avg_alg_3 = sum(alg_3) / len(alg_3)

    ret_list = []
    ret_list.append((abs(avg_alg_1 - avg_alg_2) > 2.343 * math.sqrt(1/5), "knn vs svm"))
    ret_list.append((abs(avg_alg_1 - avg_alg_3) > 2.343 * math.sqrt(1/5), "knn vs DT"))
    ret_list.append((abs(avg_alg_2 - avg_alg_3) > 2.343 * math.sqrt(1/5), "svm vs DT"))

    return ret_list

if __name__ == "__main__":
    """Function to test the friedman function"""
    knn = [0.6809, 0.7017, 0.7012, 0.6913, 0.6333, 0.6415, 0.7216, 0.7214, 0.6578, 0.7865]
    svm = [0.7164, 0.8883, 0.8410, 0.6825, 0.7599, 0.8479, 0.7012, 0.4959, 0.9279, 0.7455]
    DT = [0.7524, 0.8964, 0.6803, 0.9102, 0.7758, 0.8154, 0.6224, 0.7585, 0.938, 0.7524]
    print(friedman(knn, svm, DT))
    print(critical_difference(knn, svm, DT))
