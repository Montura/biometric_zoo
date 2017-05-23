#!/usr/bin/python
# -*- coding: utf8 -*-
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import itertools as it


# Get template size
def get_dimensions(path):
    # Read data
    with open(path, 'r') as file_matrix:
        rows = file_matrix.readline()
        cols = file_matrix.readline()
    return [rows, cols]


def load_similarity_files(db_path):
    files = []
    for index, file_name in enumerate(sorted(os.listdir(db_path))):
        files.append([])
        files[index] += [file_name]
        files[index] += [db_path]
        dims = get_dimensions(db_path + file_name)
        files[index] += [float(dims[0])]
        files[index] += [float(dims[1])]
    return files


def get_top_np_pairs(db_path, genuine):
    files = []
    file_names = sorted(os.listdir(db_path))
    if genuine:
        file_names = [x for x in file_names if 'genuine' in x]
    else:
        file_names = [x for x in file_names if 'genuine' not in x]
    for index, file_name in enumerate(file_names):
        files.append([])
        files[index] += [file_name]
        files[index] += [db_path]
    return files


def get_lsm(path):
    matrix = []
    with open(path, 'r') as file:
        file.readline()
        file.readline()
        content = file.readlines()
        for index, row in enumerate(content):
            matrix.append([])
            for col in row.split(' '):
                if col != '\n':
                    matrix[index].append(float(col))
    return matrix


def read_file(path):
    tabel = []
    with open(path, 'r') as file:
        file.readline()
        content = file.readlines()
        if len(content) == 0:
            return []
        for index, row in enumerate(content):
            tabel.append([])
            row = row.split(' ')
            if len(row) == 1:
                row = row[0].split('\t')
            for col in row:
                tabel[index].append(int(col))
    return tabel


def column(some_matrix, column_index):
    return [row[column_index] for row in some_matrix]


# Get all scores for any cylinder from LSMs, that build for all comparisons
def build_rows_cylinder_scores(lsm_files, finger_number, cylinder_number, is_genuine):
    similarity_matrices = list(it.ifilter(lambda x, c=it.count(): finger_number in x[0].split('-')[0] and next(c) < 79,
                                          lsm_files))
    comparison_count = len(similarity_matrices)  # get quantity of comparisons for one cylinder ~ 79
    count = 0
    # Vector that will be contain scores from cylinder_number row from each templates comparison
    cylinder_scores = []
    # Create LSM from all comparisons and fill vector by rows
    for i in range(0, comparison_count):
        file_info = similarity_matrices[i]
        file_name = file_info[0]
        path = file_info[1] + file_info[0]
        if "feng" in file_name:
            file_name = file_name.split('feng_')[1]
        start_file = file_name.split("-")
        left_finger = float(start_file[0].replace('_', "."))
        right_finger = float(start_file[1].replace('_', "."))
        # Condition on genuine matches: 101_1 vs 101_2 ... 101_8
        if is_genuine and (np.abs(np.floor(left_finger) - np.floor(right_finger)) == 0):  # genuine scores
            # Get LSM from comparison
            lsm_matrix = get_lsm(path)
            # Put into vector scores from each LMSs row with cylinder_number
            cylinder_scores += lsm_matrix[cylinder_number]
            # Count genuine matches. There are 7 such matches. Than after processing it we can break
            count += 1
            if count == 7:
                break
        if (not is_genuine) and (np.abs(np.floor(left_finger) - np.floor(right_finger)) != 0):  # impostor scores
            # Get LSM from comparison
            lsm_matrix = get_lsm(path)
            # Put into vector scores from each LMSs row with cylinder_number
            cylinder_scores += lsm_matrix[cylinder_number]
    return cylinder_scores


def get_genuine_matches(similarity_matrix_data, consolidation_data, parallel_algorithm_data):
    # fingers - set of fingerprint numbers 101_1, 101_2, ..., 110_7, 110_8
    fingers = set([])
    # genuine_*** - files, that help to work only with genuine matches
    genuine_lsm_files = []
    genuine_yd_files = []
    genuine_lss_files = []
    for i in range(0, len(similarity_matrix_data)):
        # Get path to some Local Similarity Matrix
        file_name = similarity_matrix_data[i][0]
        if "feng" in file_name:
            file_name = file_name.split('feng_')[1]
        # Split path name in two parts: ("101_1-101_2" -> 101.1 and 101.2)
        start_file = file_name.split("-")
        left_finger = float(start_file[0].replace('_', "."))
        right_finger = float(start_file[1].replace('_', "."))
        # Push finger number to set of fingerprint numbers
        fingers.add(start_file[0])
        # Add only genuine matches.
        # For finger print 101_1 genuine matches: 101_2, 101_3, 101_4, 101_5, 101_6, 101_7, 101_8
        if np.abs(np.floor(left_finger) - np.floor(right_finger)) == 0:
            genuine_lsm_files.append(similarity_matrix_data[i])
            genuine_yd_files.append(parallel_algorithm_data[i])
            genuine_lss_files.append(consolidation_data[i])
    return sorted(list(fingers)), genuine_lsm_files, genuine_yd_files, genuine_lss_files


def get_data_per_finger(finger_number, similarity_matrix_data, consolidation_data, parallel_algorithm_data):
    # Get cylinders quantity for current finger
    cylinders_quantity = int(next(x[2] for x in similarity_matrix_data if finger_number in x[0].split('-')[0]))

    # Print for debug
    print(finger_number)

    # Get files for current finger
    # YD_files contains pairs (positions of genuine scores in cylinders)
    yd_for_finger = list(it.ifilter(lambda x, c=it.count(): finger_number in x[0].split('-')[0] and next(c) < 8,
                                    parallel_algorithm_data))
    # MCC_files contains cylinders quantity for each matching
    lsm_for_finger = list(it.ifilter(lambda x, c=it.count(): finger_number in x[0].split('-')[0] and next(c) < 8,
                                     similarity_matrix_data))
    # LSS_files contains top lss pairs for each matching
    lss_for_finger = list(it.ifilter(lambda x, c=it.count(): finger_number in x[0].split('-')[0] and next(c) < 8,
                                     consolidation_data))

    return cylinders_quantity, lsm_for_finger, lss_for_finger, yd_for_finger


def get_genuine_scores(cylinder_number, genuine_cylinder, parallel_files, consolidation_files, lsm_files):
    # Shift in vector of scores (in some_cylinder)
    shift = 0
    cylinder_number += 1  # j + 1 because of numbers in read files from 1
    genuine_scores = []
    for k in range(0, len(parallel_files)):
        # Get numbers of genuine cylinders
        curr_file = parallel_files[k][1] + parallel_files[k][0]
        genuine_numbers = read_file(curr_file)
        lss_file = consolidation_files[k][1] + consolidation_files[k][0]
        top_lss_numbers = read_file(lss_file)
        if len(genuine_numbers) > 0:
            intersection = [x for x in top_lss_numbers for y in genuine_numbers if x == y]
            left_intersect_column = column(intersection, 0)
            if cylinder_number in left_intersect_column:
                # Get genuine score position for cylinder j
                genuine_score_number = [x[1] for x in intersection if x[0] == cylinder_number]
                for number in genuine_score_number:
                    # Calc genuine score position in cylinder (+ shift)
                    genuine_score_position = int(shift + number) - 1
                    # Add genuine score to result genuine vector
                    genuine_scores += [genuine_cylinder[genuine_score_position]]
            # Recalc shift
            shift = shift + lsm_files[k][3]
    return genuine_scores


def process_consolidation(similarity_matrices, consolidation_path):
    # Group files by type
    top_consolidation = get_top_np_pairs(consolidation_path, 0)
    top_parallel = get_top_np_pairs(consolidation_path, 1)
    # Select files: Local Similarity Matrices, Parallel_lines_algorithm, Consolidation_results
    # that contains only genuine matches in database
    fingers_set, lsm_data, parallels_data, consolidation_data = \
        get_genuine_matches(similarity_matrices, top_consolidation, top_parallel)

    # Means:
    # table with columns (fingerprint, cylinder_number, genuine_mean, impostor_mean, genuine_median, impostor_median)
    scores_table = []
    # Main cycle
    for finger_number in fingers_set:
        # Get fingerprint number from set
        # finger_number = fingers_set[i]
        cylinders_quantity, lsm_files, consolidation_files, parallel_files = \
            get_data_per_finger(finger_number, lsm_data, consolidation_data, parallels_data)
        # For current finger calc its genuine and impostor mean
        for cylinder_number in range(0, cylinders_quantity):
            # Build cylinder, taking j row from each genuine match
            genuine_scores = build_rows_cylinder_scores(similarity_matrices, finger_number, cylinder_number, 1)
            impostor_scores = build_rows_cylinder_scores(similarity_matrices, finger_number, cylinder_number, 0)

            filter_genuine_scores = \
                get_genuine_scores(cylinder_number, genuine_scores, parallel_files, consolidation_files, lsm_files)
            # Add genuine score to result genuine vector

            if len(filter_genuine_scores) == 0:
                filter_genuine_scores = [-1]
            # Save results
            scores_table.append([finger_number, cylinder_number + 1,
                                 st.mean(filter_genuine_scores), st.mean(impostor_scores),
                                 st.median(filter_genuine_scores), st.median(impostor_scores)])
        break
    return scores_table


end_lsm = ["LSM_DB2000_DB1", "LSM_DB2000_DB2", "LSM_DB2000_DB3", "LSM_DB2000_DB4",
           "LSM_DB2002_DB1", "LSM_DB2002_DB2", "LSM_DB2002_DB3", "LSM_DB2002_DB4",
           "LSM_DB2004_DB1", "LSM_DB2004_DB2", "LSM_DB2004_DB3", "LSM_DB2004_DB4"]

lss_part = "LSS/"
lss_r_part = "LSS-R/"
lsa_part = "LSA/"
lsa_r_part = "LSA-R/"
end_consolidation = ["DB2000_DB1", "DB2000_DB2", "DB2000_DB3", "DB2000_DB4",
                     "DB2002_DB1", "DB2002_DB2", "DB2002_DB3", "DB2002_DB4",
                     "DB2004_DB1", "DB2004_DB2", "DB2004_DB3", "DB2004_DB4"]

for i in range(0, len(end_lsm)):
    lsm_path = "/home/montura/R/MCC/LSM/"
    consolidation = "/home/montura/R/MCC/Consolidation/"
    # lsm_path = "/home/montura/R/Feng/LSM/"
    # consolidation = "/home/montura/R/Feng/Consolidation/"
    lsm_path = lsm_path + end_lsm[i] + '/'
    lss_path = consolidation + lss_part + end_consolidation[i] + '/'
    lss_r_path = consolidation + lss_r_part + end_consolidation[i] + '/'
    lsa_path = consolidation + lsa_part + end_consolidation[i] + '/'
    lsa_r_path = consolidation + lsa_r_part + end_consolidation[i] + '/'
    print(lsm_path)
    print(lss_path)
    print(lss_r_path)
    print(lsa_path)
    print(lsa_r_path)
    similarity_matrices_data = load_similarity_files(lsm_path)
    lss = process_consolidation(similarity_matrices_data, lss_path)
    print(lss)
    # write_scores_to_file(lss, "lss", end_consolidation[i])
    # lss_r = process_consolidation(similarity_matrices_data, lss_r_path)
    # write_scores_to_file(lss_r, "lss_r", end_consolidation[i])
    # lsa = process_consolidation(similarity_matrices_data, lsa_path)
    # write_scores_to_file(lsa, "lsa", end_consolidation[i])
    # lsa_r = process_consolidation(similarity_matrices_data, lsa_r_path)
    # write_scores_to_file(lsa_r, "lsa_r", end_consolidation[i])