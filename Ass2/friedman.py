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
    """Returns the friedman test statistic"""

    for i in range(len(knn)):
        ranked = row_ranker([knn[i], svm[i], DT[i]])
        knn[i] = ranked[0]
        svm[i] = ranked[1]
        DT[i] = ranked[2]

    avg_rank = (average(knn) + average(svm) + average(DT)) / 3
    return 10 * ((average(knn) - avg_rank) ** 2 + (average(svm) - avg_rank) ** 2 + (average(DT) - avg_rank) ** 2)