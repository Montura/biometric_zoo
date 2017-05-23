import csv
import statistics as st
from pylab import *
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
    # data = np.asarray([x for x in data if x[2] != -1.0])
    return np.asarray(data)[:, [0, 2, 3]]


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


def get_FAR_FRR(scores, threshold):
    genuine_scores = scores[:, 1]
    impostor_scores = scores[:, 2]
    impostor_count = 0
    genuine_count = 0
    for x in impostor_scores:
        if x > threshold:
            impostor_count += 1

    for x in genuine_scores:
        if x < threshold and x != -1:
            genuine_count += 1

    FAR = float(impostor_count) / float(len(impostor_scores))
    FRR = float(genuine_count) / float(len(genuine_scores))
    return FAR, FRR


consolidation = ["LSS/", "LSS-R/", "LSA/", "LSA-R/"]
first_part = ["lss_", "lss_r_", "lsa_", "lsa_r_"]
databases = ["DB2000_DB1", "DB2000_DB2", "DB2000_DB3", "DB2000_DB4",
                   "DB2002_DB1", "DB2002_DB2", "DB2002_DB3", "DB2002_DB4",
                   "DB2004_DB1", "DB2004_DB2", "DB2004_DB3", "DB2004_DB4"]
third_part = ".txt"
methods = ["mean", "median"]
method = ["mean_scores_", "median_scores_"]
threshold_values = np.arange(0.1, 1.0, 0.001)


def calculate_classes_count_global(path, method_name):
    for i in range(0, len(databases)):
        FAR = []
        FRR = []
        file_name = path + consolidation[0] + first_part[0] + databases[i] + third_part
        data = read_data(file_name)
        scores = get_scores_from_data(data, methods[0])
        max_impostor = max(scores[:, 2])
        min_genuine = min([x for x in scores[:, 1] if x != -1])
        for threshold in threshold_values:
            curr_FAR, curr_FRR = get_FAR_FRR(scores, threshold)
            FAR.append(curr_FAR)
            FRR.append(curr_FRR)
        process_data_method = method[0] + databases[i]
        plot(max_impostor, min_genuine, FAR, FRR, method_name, process_data_method)
    # print(res)


def scatter_plot_for_local_similarities(path_to_scores, method_name):
    for i in range(0, len(databases)):
        FAR = []
        FRR = []
        file_name = path_to_scores + method[0] + first_part[0] + databases[i]
        data = read_cylinders_data(file_name)
        scores = process_cylinders_data(data)
        max_impostor = max(scores[:, 2])
        min_genuine = min([x for x in scores[:, 1] if x != -1])
        for threshold in threshold_values:
            curr_FAR, curr_FRR = get_FAR_FRR(scores, threshold)
            FAR.append(curr_FAR)
            FRR.append(curr_FRR)
        process_data_method = method[0] + first_part[0] + databases[i]
        plot(max_impostor, min_genuine, FAR, FRR, method_name, process_data_method)


def plot(max_impostor, min_genuine, far, frr, algorithm, process_data_method):
    fig, ax = plt.subplots()
    ax.plot(far, frr, 'r')

    # min of both axes and max of both axes
    lims = [ np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    xlabel('FAR')
    ylabel('FRR')
    title(algorithm + ", I_max:" + str(max_impostor) + ", G_min:" + str(min_genuine))
    # plt.savefig(algorithm + "_scatter_plot_for_" + process_data_method)


path_to_mcc = "/home/montura/yandexDisk/Projects/R/LocalYagerDunstone/MCC/"
path_to_feng = "/home/montura/yandexDisk/Projects/R/LocalYagerDunstone/Feng/"


scatter_plot_for_local_similarities(path_to_mcc, "MCC")
scatter_plot_for_local_similarities(path_to_feng, "Feng")


#
path_to_global_mcc = "/home/montura/R/MCC/Consolidation/"
path_to_global_feng = "/home/montura/R/Feng/Consolidation/"

# calculate_classes_count_global(path_to_global_mcc, "MCC")
# calculate_classes_count_global(path_to_global_feng, "Feng")

#

