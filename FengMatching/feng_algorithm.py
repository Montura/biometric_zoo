#!/usr/bin/python
# -*- coding: utf8 -*-
from __future__ import print_function
import numpy as np
import cv2
import copy
import math
import sys
import os
from munkres import Munkres


class Minutia:
    def __init__(self, coord_x, coord_y, minutia_angle):
        self.x = coord_x
        self.y = coord_y
        self.angle = minutia_angle

    def __repr__(self):
        return str(1)

    def __str__(self):
        return str(str(self.x) + " " + str(self.y) + " " + str(self.angle))


def calc_euclid_distance(minutia_one, minutia_two):
    return np.sqrt(np.power((minutia_one.x - minutia_two.x), 2) + np.power((minutia_one.y - minutia_two.y), 2))


def read_minutiae_from_file(path_to_file):
    minutiae_list = []
    with open(path_to_file) as mcc_template_file:
        file_content = mcc_template_file.readlines()
        file_content = file_content[4:]
        for data in file_content:
            data = data.split(' ')
            if len(data) < 3:  # data != (x, y, angle). At this position we have read all minutiae, so we should break
                break
            minutiae_list.append(Minutia(int(data[0]), int(data[1]), float(data[2])))
    return minutiae_list


def calc_fix_radius_descriptors(template, radius):
    descriptors = []
    fixed_corrected_radius = 0.9 * radius
    for i in range(0, len(template)):
        curr_descriptor = [template[i]]  # create descriptor wtih center in i-th minutiae template
        for j in range(0, len(template)):
            if i == j:  # don't append itself to neighbours
                continue
            if calc_euclid_distance(template[i], template[j]) <= fixed_corrected_radius:  # calc fix-based neighbours
                curr_descriptor.append(template[j])
        # append built descriptor to descriptors set for current template
        descriptors.append(curr_descriptor)
    return descriptors


def rotate_descriptor(descriptor):
    rotated_descriptor = copy.deepcopy(descriptor)
    # Get descriptor center
    center_minutia = rotated_descriptor[0]
    # Get angle
    theta = center_minutia.angle
    # Get shift to origin
    left_vector = np.array([0 - center_minutia.x, 0 - center_minutia.y])  # params for shift

    # Counter-clockwise rotate matrix
    # cos(theta)  -sin(theta)
    # sin(theta) cost(theta)
    transformation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotate and shift CS
    for minutia in rotated_descriptor:
        # shift every minutia to distance to origin
        x = minutia.x + left_vector[0]
        y = minutia.y + left_vector[1]
        # rotate with angle theta
        x, y = np.matmul(transformation_matrix, np.array([x, y]))
        minutia.x, minutia.y = int(x - left_vector[0]), int(y - left_vector[1])
        # print(minutia)
    return rotated_descriptor


def paint_descriptor_on_coordinate_system(descriptor, tmp_image, before, image_name):
    for minutia in descriptor:
        if before:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.circle(tmp_image, (minutia.x, minutia.y), 2, color, thickness=3)
        cv2.putText(tmp_image, str(minutia.num), (minutia.x, minutia.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Построим равнобедренный треугольник АВС c основанием BC
        #      *(A)
        #     *  *
        # (B)******(C)
        # A(0,0); B(x2,0); C(x,y)
        # AB = 2, AC = 2, BC = ?
        ab = 1000
        ac = 1000
        flag = 1
        # Т.к. равнобедренном треугольнике высота, биссектриса и медиана - одно тоже, найдём основание треугольника
        if np.pi - minutia.angle > 0:
            betta = (np.pi - minutia.angle) / 2  # найдём величину угла при основании
        else:
            betta = (2 * np.pi - minutia.angle) / 2
            flag = -1
        bc = 2 * ac * np.cos(betta)

        # Из системы уравнений на пересечение окружностей с центрами в т.А и т.B
        # получим координаты точки C.
        # A(x1, y1), B(x2, y2), C(x, y)
        # (x - x1)^2 + (y - y1)^2 = ac^2
        # (x - x2)^2 + (y - y2)^2 = ab^2
        x2 = ab

        # # После подстановки x1 = 0, y1 = 0, x2 = 2, y2 = 0, получим две формулы
        x = round((pow(bc, 2) - pow(ac, 2) - pow(x2, 2)) / (-2 * x2), 0) * flag
        y = -np.sqrt(pow(ac, 2) - pow(x, 2)) * flag
        # Найдем уравнение прямой, которой лежит отрезок AC
        coef = 0.015
        new_x = (minutia.x + coef * (minutia.x + x)) / (1 + coef)
        new_y = (minutia.y + coef * (minutia.y + y)) / (1 + coef)
        cv2.line(tmp_image, (minutia.x, minutia.y), (int(new_x), int(new_y)), color, thickness=2)
    cv2.imshow(image_name, tmp_image)
    cv2.waitKey(0)


def filter_descriptor(descriptor, fixed_descriptor_radius):
    del_count = 0
    for i in range(1, len(descriptor)):
        i -= del_count
        distance_to_center = calc_euclid_distance(descriptor[0], descriptor[i])
        if distance_to_center > 0.9 * fixed_descriptor_radius:
            del descriptor[i]
            del_count += 1


def feng_algorithm(descriptors_for_one, descriptors_for_two):
    # Similarity matrix between descriptor's pairs
    similarity_matrix = np.zeros((len(descriptors_for_one), len(descriptors_for_two)), dtype=float)
    row_number = 0
    for d_one in descriptors_for_one:
        column_number = 0
        for d_two in descriptors_for_two:
            dx_shift = d_two[0].x - d_one[0].x
            dy_shift = d_two[0].y - d_one[0].y
            # counts for minutiae that can be matched
            mp_count_array = np.zeros(len(d_one), int)
            mq_count_array = np.zeros(len(d_two), int)
            # total minutiae number in descriptors without 1, i.e. without descriptor's center minutia
            _Mp = len(d_one) - 1
            _Mq = len(d_two) - 1
            # take some center's neighbour in first descriptor
            for k in range(1, len(d_one)):
                # append parallel shift between two descriptor's centers to its coordinates
                tmp_x = d_one[k].x + dx_shift
                tmp_y = d_one[k].y + dy_shift
                for l in range(1, len(d_two)):
                    # calc distance between shifted neighbour from first descriptor and some neighbour from second
                    distance = calc_euclid_distance(Minutia(tmp_x, tmp_y, d_one[k].angle), d_two[l])
                    # If after parallel shift two minutiae can be matchable (about 6 px between them)
                    if round(distance, 0) <= 6:
                        mp_count_array[k] = 1
                        mq_count_array[l] = 1
            mp_count = sum(mp_count_array)
            mq_count = sum(mq_count_array)
            # similarity_value = (mp_count + 1) / float(Mp + 1) * (mq_count + 1) / float(Mq + 1)
            similarity_matrix[row_number][column_number] = (mp_count + 1) / float(_Mp + 1) * (mq_count + 1) / float(_Mq + 1)
            column_number += 1
        row_number += 1
    return similarity_matrix


def new_image():
    image = np.full((500, 500, 3), 255, np.uint8)
    height, width, channels = image.shape
    for col in range(0, width, 10):
        cv2.line(image, (col, 0), (col, height), (0, 0, 0))
    for row in range(0, height, 10):
        cv2.line(image, (0, row), (width, row), (0, 0, 0))
    return image


def read_local_similarity_matrix(path):
    with open(path) as local_similarities_file:
        content = local_similarities_file.readlines()
        rows = int(content[0])
        columns = int(content[1])
        content = content[2:]
        matrix = np.zeros((rows, columns), dtype=float)
        for i in range(0, rows):
            content[i] = content[i].split(" ")
            for j in range(0, columns):
                matrix[i][j] = float(content[i][j])
    return matrix


def prepare_for_lss(some_similarity_matrix):
    array_for_lss = []
    for i in range(0, len(some_similarity_matrix)):
        for j in range(0, len(some_similarity_matrix[i])):
            array_for_lss.append([some_similarity_matrix[i][j], i, j])
    return sorted(array_for_lss, key=lambda x: x[0], reverse=True)


def sigmoid_function(v, nju, tau):
    if abs(-tau * (v - nju)) > 100:
        v = 100 / -tau + nju
    return 1 / float(1 + np.exp(-tau * (v - nju)))


def calc_parameter_np_for_lss(_na, _nb):
    _min_np = 4
    _max_np = 12
    _nju = 20
    _tau = 0.4
    v = min(_na, _nb)
    sigmoid_result = sigmoid_function(v, _nju, _tau)
    return int(_min_np + round(sigmoid_result * (_max_np - _min_np), 0))


def lss(_np, sorted_array_for_lss):
    result = []
    top_np_similarities = sorted_array_for_lss[:_np]
    for element in top_np_similarities:
        result.append([element[1], element[2]])
        # print([element[1], element[2]])
    return result


def calc_diff_phi(theta_one, theta_two):
    angles_diff = theta_one - theta_two
    if -np.pi <= angles_diff < np.pi:
        return angles_diff
    if angles_diff < -np.pi:
        return 2 * np.pi + angles_diff
    if angles_diff >= np.pi:
        return -2 * np.pi + angles_diff


def calc_diff_theta(minutia_one, minutia_two):
    return calc_diff_phi(minutia_one.angle, minutia_two.angle)


# radial angle is defined as
# the angle subtended by the edge connecting the two minutiae and the direction of the first one
def calc_radial_angle(minutia_one, minutia_two):
    atan2 = math.atan2(-minutia_two.y + minutia_one.y, minutia_two.x - minutia_one.x)
    return calc_diff_phi(minutia_one.angle, atan2)


# rho = p(t, k) - measure of the compatibility between two pairs of minutiae:
# minutiae (a_rt, a_rk) of template A and minutiae (b_ct, b_ck) of template B
def calc_rho(t, k, sorted_array_for_lss, templ_a, templ_b):
    # Get minutiae positions in matrix Gamma
    _rt = sorted_array_for_lss[t][1]
    _ct = sorted_array_for_lss[t][2]
    _rk = sorted_array_for_lss[k][1]
    _ck = sorted_array_for_lss[k][2]

    # Calc euclid distances ds(a_rt, a_rk), ds(b_ct, a_ck)
    euclid_dist_for_minutiae_in_a = calc_euclid_distance(templ_a[_rt], templ_a[_rk])
    euclid_dist_for_minutiae_in_b = calc_euclid_distance(templ_b[_ct], templ_b[_ck])
    # d 1 denotes the similarity between the minutiae spatial distances
    d1 = abs(euclid_dist_for_minutiae_in_a - euclid_dist_for_minutiae_in_b)

    # Calc the directional difference between two minutiae
    directional_diff_for_minutiae_in_a = calc_diff_theta(templ_a[_rt], templ_a[_rk])
    directional_diff_for_minutiae_in_b = calc_diff_theta(templ_b[_ct], templ_b[_ck])
    # d2 compares the directional differences
    d2 = abs(calc_diff_phi(directional_diff_for_minutiae_in_a, directional_diff_for_minutiae_in_b))

    # Radial angle is defined as the angle subtended by the edge connecting the two minutiae
    # and the direction of the first one
    radial_angles_diff_for_minutiae_in_a = calc_radial_angle(templ_a[_rt], templ_a[_rk])
    radial_angles_diff_for_minutiae_in_b = calc_radial_angle(templ_b[_ct], templ_b[_ck])
    # d3 compares the radial angles
    d3 = abs(calc_diff_phi(radial_angles_diff_for_minutiae_in_a, radial_angles_diff_for_minutiae_in_b))

    # Parameters for Sigmoid function for d1,d2,d3
    nju_p_1, tau_p_1 = 5, -8.0 / 5.0
    nju_p_2, tau_p_2 = np.pi / 12, -30
    nju_p_3, tau_p_3 = np.pi / 12, -30

    # Calc Sigmoid function for d1,d2,d3
    sigmoid_1 = sigmoid_function(d1, nju_p_1, tau_p_1)
    sigmoid_2 = sigmoid_function(d2, nju_p_2, tau_p_2)
    sigmoid_3 = sigmoid_function(d3, nju_p_3, tau_p_3)
    return sigmoid_1 * sigmoid_2 * sigmoid_3


def lss_r(_na, _nb, _np, sorted_array_for_lss, template_one, tempate_two):
    _nR = min(_na, _nb)  # t = 0...._nR
    _nRel = 5  # number for relaxations for LSS-R
    _wR = 0.5  # weight parameter
    lambdas = np.zeros((_nRel + 1, _nR), float)  #
    count = 0
    top_nR_similarity = sorted_array_for_lss[:_nR]
    for element in top_nR_similarity:  # top nR similarity values from Gamma[r,c];
        if element != 0.0:
            lambdas[0][count] = element[0]  # lambdas[0][t] = G[r_t, c_t] - the initial similarity of pair t;
        else:
            lambdas[0][count] = 1
        count += 1
    # the similarity at iterations
    for i in range(1, _nRel + 1):
        for t in range(0, _nR):
            _total_Rho = 0.0
            for k in range(0, _nR):
                _total_Rho += calc_rho(t, k, top_nR_similarity, template_one, tempate_two) * lambdas[i - 1][k]
            # the similarity at iteration i of the relaxation procedure is
            lambdas[i][t] = _wR * lambdas[i - 1][t] + (1 - _wR) * _total_Rho / (_nR - 1)
            # check
            if lambdas[i][t] > 1:
                print("ERROR")
    # efficiency of pair t is calculated as  Eps[t] = lambdas[_nRel][t] / lambdas[0][t]
    epsilon_efficiency = []
    for t in range(0, len(lambdas[0])):
        epsilon_efficiency.append([lambdas[_nRel][t] / lambdas[0][t], t])
    top_np_lsr_efficiency = sorted(epsilon_efficiency, key=lambda x: x[0], reverse=True)[:_np]
    top_np_similarities = []
    for element in top_np_lsr_efficiency:
        top_np_similarities.append(top_nR_similarity[element[1]][1:3])
    return top_np_similarities


def lsa(_np, similarity_matrix):
    multiply_value = 10 ** math.floor(np.log10(sys.maxsize))
    cost_matrix = []
    for row in similarity_matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - int(col * multiply_value)]
        cost_matrix += [cost_row]
    m = Munkres()
    indexes_for_max = m.compute(cost_matrix)
    # total = 0
    top_np_values_for_lssr = []
    for row, column in indexes_for_max:
        value = similarity_matrix[row][column]
        # total += value
        if value == 0.0:
            value = 1 / 10 ** math.floor(np.log10(sys.maxsize))
        top_np_values_for_lssr.append([value, row, column])
    sorted_top_np_values_for_lssr = sorted(top_np_values_for_lssr, key=lambda x: x[0], reverse=True)
    return sorted_top_np_values_for_lssr[:_np], sorted_top_np_values_for_lssr


def write_lss_top_np_pairs_to_file(path_to_dir, first_file, second_file, genuine_indices, method):
    split_name = first_file.rsplit('_')
    first_name = split_name[0] + "_" + split_name[1]
    split_name = second_file.rsplit('_')
    second_name = split_name[0] + "_" + split_name[1]
    file_name = path_to_dir + first_name + "-" + second_name + ".txt"
    print(file_name)
    with open(file_name, 'w') as f:
        f.write("row_number" + "\tcolumn_number" + "\n")
        for indices in genuine_indices:
            if method == "LSA":
                f.write(str(indices[1] + 1) + "\t" + str(indices[2] + 1) + "\n")  # for LSA
            else:
                f.write(str(indices[0] + 1) + "\t" + str(indices[1] + 1) + "\n")  # for LSS|LSS-R|LSA-R


def write_similarity_matrix_to_file(path_to_dir, first_file, second_file, some_similarity_matrix):
    split_name = first_file.rsplit('_')
    first_name = split_name[0] + "_" + split_name[1]
    split_name = second_file.rsplit('_')
    second_name = split_name[0] + "_" + split_name[1]
    file_name = path_to_dir + "feng_" + first_name + "-" + second_name
    with open(file_name, 'w') as f:
        f.write(str(len(some_similarity_matrix)) + '\n')
        f.write(str(len(some_similarity_matrix[0])) + '\n')
        for row in some_similarity_matrix:
            for column in row:
                f.write(str(column) + " ")
            f.write("\n")


def create_lsm_matrix_for_feng(path_to_templates, databases_list, end_of_path, path_to_lsm_files, lss_dirs_list):
    fixed_descriptor_radius = 60.0
    for i in range(0, len(databases_list)):
        full_database_path = path_to_templates + databases_list[i] + end_of_path
        print(full_database_path)
        full_lsm_path = path_to_lsm_files + lss_dirs_list[i] + "/"
        print(full_lsm_path)
        for firstFile in sorted(os.listdir(full_database_path)):
            path_to_file = full_database_path + firstFile
            template_one = read_minutiae_from_file(path_to_file)
            descriptors_for_one = calc_fix_radius_descriptors(template_one, fixed_descriptor_radius)
            for i in range(0, len(descriptors_for_one)):
                descriptors_for_one[i] = rotate_descriptor(descriptors_for_one[i])
            for secondFile in sorted(os.listdir(full_database_path)):
                if firstFile == secondFile:
                    continue
                path_to_file = full_database_path + secondFile
                template_two = read_minutiae_from_file(path_to_file)
                descriptors_for_two = calc_fix_radius_descriptors(template_two, fixed_descriptor_radius)
                for i in range(0, len(descriptors_for_two)):
                    descriptors_for_two[i] = rotate_descriptor(descriptors_for_two[i])
                similarity_matrix = feng_algorithm(descriptors_for_one, descriptors_for_two)
                write_similarity_matrix_to_file(full_lsm_path, firstFile, secondFile, similarity_matrix)
                print(firstFile, secondFile, sep=" ")


def calc_consolidation(templates, end_templates, end_of_path, lsm, end_lsm,
                       consolidation, lss_part, lss_r_part, lsa_part, lsa_r_part, end_consolidation):
    for i in range(0, len(end_templates)):
        template_path = templates + end_templates[i] + end_of_path
        database_path = lsm + end_lsm[i] + "/"
        lss_path = consolidation + lss_part + end_consolidation[i] + "/"
        lss_r_path = consolidation + lss_r_part + end_consolidation[i] + "/"
        lsa_path = consolidation + lsa_part + end_consolidation[i] + "/"
        lsa_r_path = consolidation + lsa_r_part + end_consolidation[i] + "/"
        print(database_path)
        print(template_path)
        print(lss_path)
        print(lss_r_path)
        print(lsa_path)
        print(lsa_r_path)
        for matrix in sorted(os.listdir(database_path)):
            similarity_matrix = read_local_similarity_matrix(database_path + matrix)
            sorted_array_for_lss = prepare_for_lss(similarity_matrix)
            na = len(similarity_matrix)
            nb = len(similarity_matrix[0])
            templates_name = matrix.split("feng_")[1].split('-')  # For Feng
            # For MCC templates_name = matrix.split('-')
            path_to_first_template = template_path + templates_name[0] + "_ist.txt"
            first_file = templates_name[0] + "_ist.txt"
            path_to_second_template = template_path + templates_name[1] + "_ist.txt"
            second_file = templates_name[1] + "_ist.txt"
            template_one = read_minutiae_from_file(path_to_first_template)
            template_two = read_minutiae_from_file(path_to_second_template)
            _np = calc_parameter_np_for_lss(na, nb)
            # break
            # LSS
            top_np_lss = lss(_np, sorted_array_for_lss)
            write_lss_top_np_pairs_to_file(lss_path, first_file, second_file, top_np_lss, "LSS")
            # LSS_R
            top_np_lss_r = lss_r(na, nb, _np, sorted_array_for_lss, template_one, template_two)
            write_lss_top_np_pairs_to_file(lss_r_path, first_file, second_file, top_np_lss_r, "LSS-R")
            # LSA
            top_np_lsa, top_nr_for_lsa_r = lsa(_np, similarity_matrix)
            write_lss_top_np_pairs_to_file(lsa_path, first_file, second_file, top_np_lsa, "LSA")
            # LSA_R
            top_np_lsa_r = lss_r(na, nb, _np, top_nr_for_lsa_r, template_one, template_two)
            write_lss_top_np_pairs_to_file(lsa_r_path, first_file, second_file, top_np_lsa_r, "LSA-R")


# -------------------------------------- Local Similarity Matrices for Feng --------------------------------------
templates = "/media/montura/Media/Databases/DB_templates/"
end_templates = ["DB_2000/DB1_B", "DB_2000/DB2_B", "DB_2000/DB3_B", "DB_2000/DB4_B", "DB_2002/DB1_B", "DB_2002/DB2_B",
                 "DB_2002/DB3_B", "DB_2002/DB4_B", "DB_2004/DB1_B", "DB_2004/DB2_B", "DB_2004/DB3_B", "DB_2004/DB4_B"]
end_of_path = "/templates/"

lsm = "/home/montura/R/Feng/LSM/"
end_lsm = ["LSM_DB2000_DB1", "LSM_DB2000_DB2", "LSM_DB2000_DB3", "LSM_DB2000_DB4", "LSM_DB2002_DB1", "LSM_DB2002_DB2",
           "LSM_DB2002_DB3", "LSM_DB2002_DB4", "LSM_DB2004_DB1", "LSM_DB2004_DB2", "LSM_DB2004_DB3", "LSM_DB2004_DB4"]

# create_lsm_matrix_for_feng(templates, end_templates, end_of_path, lsm, end_lsm)

# -------------------------------------- LSS, LSS_R, LSA, LSA_R for some LSM --------------------------------------
consolidation = "/home/montura/R/Feng/Consolidation/"
lss_part = "LSS/"
lss_r_part = "LSS-R/"
lsa_part = "LSA/"
lsa_r_part = "LSA-R/"
end_consolidation = ["DB2000_DB1", "DB2000_DB2", "DB2000_DB3", "DB2000_DB4", "DB2002_DB1", "DB2002_DB2", "DB2002_DB3",
                     "DB2002_DB4", "DB2004_DB1", "DB2004_DB2", "DB2004_DB3", "DB2004_DB4"]

calc_consolidation(templates, end_templates, end_of_path, lsm, end_lsm, \
                   consolidation, lss_part, lss_r_part, lsa_part, lsa_r_part, end_consolidation)
