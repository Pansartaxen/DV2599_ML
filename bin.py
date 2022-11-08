def discrete_transform(spam: pd.DataFrame, nonspam: pd.DataFrame):
    cols = spam.columns.values.tolist()
    discrete = pd.DataFrame(columns=cols)
    discrete.loc[0] = False

    for col in range(spam.shape[1]):
        spam_col_sum = 0
        nonspam_col_sum = 0
        spam_count = 0
        nonspam_count = 0

        for row in range(spam.shape[0]):
            if spam.iloc[row,col] in [1,2,3]:
                spam_col_sum += spam.iloc[row,col]
                spam_count += 1
            if nonspam.iloc[row,col] in [1,2,3]:
                nonspam_col_sum += nonspam.iloc[row,col]
                nonspam_count += 1

        spam_mean = spam_col_sum / spam_count
        nonspam_mean = nonspam_col_sum / nonspam_count

        if int(spam_mean) != int(nonspam_mean) and spam_mean > 1:
            discrete.iloc[0,col] = True
            print(cols[col])

    # for col in spam.columns:
    #     if spam[col] != nonspam[col] and spam[col] != "low":
    #         print(col)
    #         discrete[col] = True
    return discrete

def above_zero(data: pd.DataFrame):
    words = []
    for col in data.columns:
        cur = data[col].sort_values(ascending=False)[:int(data.shape[0]*0.70)]
        if cur.min() > 0:
            words.append(col)
    return words

def get_diff(data: pd.DataFrame, data2: pd.DataFrame):
    words2 = above_zero(data)
    words1 = above_zero(data2)
    words = [col for col in words1 if col not in words2]
    return words