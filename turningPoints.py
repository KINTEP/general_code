import pandas as pd
import numpy as np

def normalize(sequence):
    max1 = np.max(sequence)
    min1 = np.min(sequence)
    data = []
    for val in sequence:
        if max1 - min1 == 0:
            data.append(0)
        else:
            norm = (val - min1)/(max1 - min1)
            data.append(norm)
    return data

def turning_points(seq, wind = 5):
    start = 0
    end = wind

    data = {}
    N = len(seq) - wind
    for i in range(N):
        l1 = seq[start:end+1]
        l2 = seq[end:end+wind+1]
        if len(l2) < len(l1):
            break
        if max(l1) == l1[-1] and max(l2) == l2[0]:
            data[end] = l1[-1]
        if min(l1) == l1[-1] and min(l2) == l2[0]:
            data[end] = l1[-1]
        start += 1
        end += 1
    #This portion was put into a dataframe so i could drop the duplicate values easily
    df2 = pd.DataFrame()
    df2["values"] = data.values()
    df2["keys"] = data.keys()
    df2.drop_duplicates(subset = "values", keep = "first", inplace = True)
    return df2["keys"], df2["values"]

def make_norm(df_keys, seq):
    latta = list(df_keys)
    nana1 = len(latta)
    norm = normalize(seq[:latta[0]])
    for i in range(nana1-1):
        l1 = normalize(seq[latta[i]:latta[i+1]])
        norm += l1
    norm += normalize(seq[latta[-1]:])
    return norm

def get_tp(seq, wind):
    TP, _ = turning_points(wind = wind, seq = seq)
    norm = make_norm(df_keys = TP, seq = seq)
    return norm