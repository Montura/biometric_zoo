#!/usr/bin/python
# -*- coding: utf8 -*-
import csv
import statistics as st
from pylab import *
from scipy import stats
from astropy.table import Table, Column
from astropy.io import ascii
import numpy as np


def read_data(path):
    array = []
    with open(path, 'r') as csvfile:
        content = csv.reader(csvfile, delimiter=' ')
        for row in content:
            row[0] = float(row[0].replace('_', "."))
            row[1] = float(row[1].replace('_', "."))
            row[2] = float(row[2])
            array.append(row)
    return array


def get_scores_from_data(data, method):
    result = []
    i = 0
    # There are 6320 comparisons: 79 impostors comparisons for each image (79 * 80)
    while i < len(data):
        genuine_scores = []
        impostor_scores = []
        # Get first image
        fingerprint_number = data[i][0]
        # Get all impostor scores fro this image
        scores = [x for x in data if x[0] == fingerprint_number]
        # Save result
        for x in scores:
            if np.abs(np.floor(x[1]) - np.floor(fingerprint_number)) == 0:
                genuine_scores.append(x[2])
            else:
                impostor_scores.append(x[2])
        if method == "mean":
            result.append([fingerprint_number, st.mean(genuine_scores), st.mean(impostor_scores)])
        else:
            result.append([fingerprint_number, st.median(genuine_scores), st.median(impostor_scores)])
        # Jump to impostor scores for next image
        i += 79
    return np.array(result)


def read_cylinders_data(path):
    array = []
    with open(path, 'r') as file:
        file.readline()
        content = file.readlines()
        for row in content:
            row = row.split('\t')
            row[0] = float(row[0].replace('_', "."))
            row[1] = int(row[1])
            row[2] = float(row[2])
            row[3] = float(row[3])
            array.append(row)
    return array


def column(some_matrix, column_index):
    return [row[column_index] for row in some_matrix]


def process_cylinders_data(data):
    total_cylinders = len(data)
    data = np.asarray([x for x in data if x[2] != -1.0])
    return data[:, [0, 1, 2, 3]], total_cylinders


def statistical_test(chameleon_count, phantoms_count, doves_count, worms_count, total_cylinders):
    # Табличка
    table = [[], [], [], []]
    p = 1.0 / 16.0

    table[0] = [chameleon_count, phantoms_count, doves_count, worms_count]  # Class members quantity
    # Binomial test
    for x in table[0]:
        table[1] += [stats.binom_test(x, total_cylinders, p)]
    # Check: if H0 < 0.05, than H0 rejected
    for x in table[1]:
        table[2] += [x < 0.05]
    # If (H0 < 0.05) and zoo_count > (1/16 * total_count) than class found
    for i, x in enumerate(table[0]):
        if (x > p * total_cylinders) and table[2][i]:
            table[3] += [1]
        else:
            table[3] += [-1]
    return table


def get_percentiles(scores, local):
    if local:
        genuine_scores = sorted(scores[:, 2])
        impostor_scores = sorted(scores[:, 3])
    else:
        genuine_scores = sorted(scores[:, 1])
        impostor_scores = sorted(scores[:, 2])
    # Из всех genuine_score_means выбираю 25 худших.
    # Выбираю нижнии границы для Gh и Ih
    top_25_genuine_high = min([x for x in genuine_scores if x >= np.percentile(genuine_scores, 75)])
    top_25_impostor_high = min([x for x in impostor_scores if x >= np.percentile(impostor_scores, 75)])
    # Выбираю верхние границы для Gl и Il
    bottom_25_genuine_low = max([x for x in genuine_scores if x <= np.percentile(genuine_scores, 25)])
    bottom_25_impostor_low = max([x for x in impostor_scores if x <= np.percentile(impostor_scores, 25)])
    # Chameleons Gh n Ih. Chameleons rarely cause false rejects, but are likely to cause false accepts.
    # Phantoms Gl n Il. Phantoms lead to low match scores regardless of who they are being matched against; themselves or others.
    # Doves Gh n Il. Doves pure and recognizable, matching well against themselves and poorly against others.
    # Worms Gl n Ih. Worms are the cause of a disproportionate number of a system’s errors
    return top_25_genuine_high, bottom_25_genuine_low, top_25_impostor_high, bottom_25_impostor_low


def find_bad_cylinders(scores, Gh, Ih):
    bad_cylinders = [x[0:2] for x in scores if x[2] >= Gh and x[3] >= Ih]
    return bad_cylinders


def compare_pairs(lhs, rhs):
    if np.isclose(lhs[0], rhs[0], rtol=1e-05, atol=1e-08, equal_nan=False) and lhs[1] == rhs[1]:
        return True
    else:
        return False


def filter_scores(scores, Gh, Ih):
    bad_cylinders = find_bad_cylinders(scores, Gh, Ih)
    count = 0
    filtered_scores = []
    for x in scores:
        curr_pair = [x[0], x[1]]
        if count < len(bad_cylinders):
            bad_pair = bad_cylinders[count]
            if compare_pairs(bad_pair, curr_pair):
                count += 1
                continue
        filtered_scores.append(x)
    # print bad_cylinders
    return np.asarray(filtered_scores)


def yager_dunstone_test(scores, local):
    Gh, Gl, Ih, Il = get_percentiles(scores, local)
    if local:
        scores = filter_scores(scores, Gh, Ih)
    if local:
        genuine_scores = scores[:, 2]
        impostor_scores = scores[:, 3]
    else:
        genuine_scores = scores[:, 1]
        impostor_scores = scores[:, 2]
    total_cylinders = len(scores)
    chameleon_count = 0
    phantoms_count = 0
    doves_count = 0
    worms_count = 0
    for i in range(0, len(scores)):
        # Chameleons
        if (genuine_scores[i] >= Gh) and (impostor_scores[i] >= Ih):
            chameleon_count += 1
        # Phantoms
        if (genuine_scores[i] <= Gl) and (impostor_scores[i] <= Il):
            phantoms_count += 1
        # Doves
        if (genuine_scores[i] >= Gh) and (impostor_scores[i] <= Il):
            doves_count += 1
        # Worms
        if (genuine_scores[i] <= Gl) and (impostor_scores[i] >= Ih):
            worms_count += 1

    # fig, ax = plt.subplots()
    #
    # ax.plot(scores[:, 1], scores[:, 2], 'o')
    # axhline(Il, 0, 1, color='r', label='Impostor low', linewidth=1.4)
    # axhline(Ih, 0, 1, color='g', label='Impostor high', linewidth=1.4)
    # axvline(Gh, 0, 1, color='b', label='Genuine high', linewidth=1.4)
    # axvline(Gl, 0, 1, color='k', label='Genuine low', linewidth=1.4)
    # xlabel('Impostor scores')
    # ylabel('Genuine scores')
    # title('YD_classes')
    # legend = ax.legend(loc='lower left', shadow=True)
    # plt.show()
    return statistical_test(chameleon_count, phantoms_count, doves_count, worms_count, total_cylinders)



consolidation = ["LSS/", "LSS-R/", "LSA/", "LSA-R/"]
first_part = ["lss_", "lss_r_", "lsa_", "lsa_r_"]
second_part = ["DB2000_DB1", "DB2000_DB2", "DB2000_DB3", "DB2000_DB4",
                   "DB2002_DB1", "DB2002_DB2", "DB2002_DB3", "DB2002_DB4",
                   "DB2004_DB1", "DB2004_DB2", "DB2004_DB3", "DB2004_DB4"]


def calculate_classes_count_global(res):
    path = "/home/montura/R/MCC/Consolidation/"

    third_part = ".txt"
    methods = ["mean", "median"]
    count = 0
    for i in range(0, len(consolidation)):
        for j in range(0, len(second_part)):
            file_name = path + consolidation[i] + first_part[i] + second_part[j] + third_part
            data = read_data(file_name)
            for k in range(0, len(methods)):
                scores = get_scores_from_data(data, methods[k])
                table = yager_dunstone_test(scores, len(scores), False)
                for index in range(0, 4):
                    if table[3][index] > 0:
                        res[index, count + k] += 1
        count += 2
    print(res)


def calculate_classes_count_local():
    res = np.zeros((4, 8), dtype=int)
    path_to_scores = "/home/montura/yandexDisk/Projects/R/LocalYagerDunstone/MCC/"
    # path_to_scores = "/home/montura/yandexDisk/Projects/R/LocalYagerDunstone/Feng/"
    method = ["mean_scores_", "median_scores_"]

    count = 0
    for i in range(0, len(consolidation)):
        for j in range(0, len(second_part)):
            for k in range(0, len(method)):
                file_name = path_to_scores + method[k] + first_part[i] + second_part[j]
                data = read_cylinders_data(file_name)
                scores, total_cylinders = process_cylinders_data(data)
                mean_table = yager_dunstone_test(scores, True)
                for index in range(0, 4):
                    if mean_table[3][index] > 0:
                        res[index, count + k] += 1
        count += 2
    return res

# res = np.zeros((4, 8), dtype=int)
# calculate_classes_count_global(res)
# fig, ax = plt.subplots()
# ax.plot([1, 2], [3, 4])
# xlabel('Impostor scores')
# ylabel('Genuine scores')
# title('YD_classes')
# legend = ax.legend(loc='lower left', shadow=True)
# plt.show()


res = calculate_classes_count_local()
t = Table(res, names=("LSS_mean", "LSS_median", "LSS-R_mean", "LSS-R_median",
                 "LSA_mean", "LSA_median", "LSA-R_mean", "LSA-R_median"),
          dtype=('i4', 'i4', 'i4', "i4", 'i4', 'i4', 'i4', "i4"))
col = Column(['C', 'P', 'D', 'W'], name="Classes")
t.add_column(col, index=0)
t.pprint(-1, -1)

ascii.write(t, format='latex')

# calculate_classes_count_local(res)
# t.write(format='latex')