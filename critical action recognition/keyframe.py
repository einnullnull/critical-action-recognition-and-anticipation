import csv
import math

import numpy
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from hmmlearn import hmm
import seaborn as sns


# Obtain the kernel model and the corresponding thresholds
def key_kde(x_pos, y_pos):
    xy_pos = np.vstack([x_pos, y_pos])
    kenal_p = gaussian_kde(xy_pos)
    z_pos = kenal_p.evaluate(xy_pos)
    z_pos = numpy.array(z_pos)
    lim = z_pos.mean() / 8

    return kenal_p, lim


# Get the HMM model
def hmm_model(tool_name):
    states = [0, 1]
    n_states = len(states)
    observations = [0, 1, 2, 3]
    n_observations = len(observations)
    model1 = hmm.MultinomialHMM(n_components=n_states, n_iter=50, tol=0.01)
    # Hidden status unknown
    T1 = tool_name['num_tip'].values.reshape(-1, 1).astype(int)
    model1.fit(T1)
    return model1

# Get the area size threshold
def area_lim(tool_name):
    tool_name['area'].describe()
    key_mean_area = tool_name['area'].describe()
    lim_min_area = (key_mean_area['min'] * 0.98)
    lim_max_area = (key_mean_area['max'] * 1.02)
    return lim_min_area, lim_max_area



def get_all_list():
    # Database information integration
    file_path15 = "keyframe_pre01.csv"
    df15 = pd.read_csv(file_path15)

    file_path16 = "keyframe_pre02.csv"
    df16 = pd.read_csv(file_path16)

    df = pd.concat([df15, df16], axis=0)
    df = df.dropna(axis=0, how='any', inplace=False)

    df.index = np.arange(len(df.index))

    Intestinal_forceps = df[df['class'] == "Intestinal forceps"].iloc[:, [1, 2, 3, 4, 5, 6]]
    Ligation_clip = df[df['class'] == "Ligation clip"].iloc[:, [1, 2, 3, 4, 5, 6]]
    Right_angle_separating_pliers = df[df['class'] == "Right angle separating pliers"].iloc[:, [1, 2, 3, 4, 5, 6]]
    Separating_pliers = df[df['class'] == "Separating pliers"].iloc[:, [1, 2, 3, 4, 5, 6]]
    electric_hook = df[df['class'] == "electric hook"].iloc[:, [1, 2, 3, 4, 5, 6]]
    scissors = df[df['class'] == "scissors"].iloc[:, [1, 2, 3, 4, 5, 6]]
    Bipolar_electrocoagulation = df[df['class'] == "Bipolar electrocoagulation"].iloc[:, [1, 2, 3, 4, 5, 6]]
    # ultrasound_knife = df[df['class'] == "ultrasound knife"].iloc[:, [1, 2, 3, 4, 5, 6]]

    sum_frame = [Ligation_clip, scissors, electric_hook, Right_angle_separating_pliers, Separating_pliers,
                 Bipolar_electrocoagulation, Intestinal_forceps]
    # Get dynamic information
    ac_Bipolar_electrocoagulation = Bipolar_electrocoagulation - Bipolar_electrocoagulation.shift(1)
    ac_Bipolar_electrocoagulation['keyFrame'] = Bipolar_electrocoagulation['keyFrame']

    ac_Intestinal_forceps = Intestinal_forceps - Intestinal_forceps.shift(1)
    ac_Intestinal_forceps['keyFrame'] = Intestinal_forceps['keyFrame']

    ac_Ligation_clip = Ligation_clip - Ligation_clip.shift(1)
    ac_Ligation_clip['keyFrame'] = Ligation_clip['keyFrame']

    ac_Right_angle_separating_pliers = Right_angle_separating_pliers - Right_angle_separating_pliers.shift(1)
    ac_Right_angle_separating_pliers['keyFrame'] = Right_angle_separating_pliers['keyFrame']

    ac_Separating_pliers = Separating_pliers - Separating_pliers.shift(1)
    ac_Separating_pliers['keyFrame'] = Separating_pliers['keyFrame']

    ac_electric_hook = electric_hook - electric_hook.shift(1)
    ac_electric_hook['keyFrame'] = electric_hook['keyFrame']

    ac_scissors = scissors - scissors.shift(1)
    ac_scissors['keyFrame'] = scissors['keyFrame']

    ac_Bipolar_electrocoagulation = ac_Bipolar_electrocoagulation.dropna(axis=0, how='any', inplace=False)
    # ac_ultrasound_knife = ac_ultrasound_knife.dropna(axis=0, how='any', inplace=False)
    ac_Intestinal_forceps = ac_Intestinal_forceps.dropna(axis=0, how='any', inplace=False)
    ac_Ligation_clip = ac_Ligation_clip.dropna(axis=0, how='any', inplace=False)
    ac_Right_angle_separating_pliers = ac_Right_angle_separating_pliers.dropna(axis=0, how='any', inplace=False)
    ac_Separating_pliers = ac_Separating_pliers.dropna(axis=0, how='any', inplace=False)
    ac_electric_hook = ac_electric_hook.dropna(axis=0, how='any', inplace=False)
    ac_scissors = ac_scissors.dropna(axis=0, how='any', inplace=False)

    ac_Intestinal_forceps_1 = ac_Intestinal_forceps[ac_Intestinal_forceps['keyFrame'] == 1]
    ac_Ligation_clip_1 = ac_Ligation_clip[ac_Ligation_clip['keyFrame'] == 1]
    ac_Right_angle_separating_pliers_1 = ac_Right_angle_separating_pliers[
        ac_Right_angle_separating_pliers['keyFrame'] == 1]
    ac_Separating_pliers_1 = ac_Separating_pliers[ac_Separating_pliers['keyFrame'] == 1]
    ac_electric_hook_1 = ac_electric_hook[ac_electric_hook['keyFrame'] == 1]
    ac_scissors_1 = ac_scissors[ac_scissors['keyFrame'] == 1]
    ac_Bipolar_electrocoagulation_1 = ac_Bipolar_electrocoagulation[ac_Bipolar_electrocoagulation['keyFrame'] == 1]
    # ac_ultrasound_knife_1 = ac_ultrasound_knife[ac_ultrasound_knife['keyFrame'] == 1]

    Intestinal_forceps_1 = Intestinal_forceps[Intestinal_forceps['keyFrame'] == 1]
    Ligation_clip_1 = Ligation_clip[Ligation_clip['keyFrame'] == 1]
    Right_angle_separating_pliers_1 = Right_angle_separating_pliers[Right_angle_separating_pliers['keyFrame'] == 1]
    Separating_pliers_1 = Separating_pliers[Separating_pliers['keyFrame'] == 1]
    electric_hook_1 = electric_hook[electric_hook['keyFrame'] == 1]
    scissors_1 = scissors[scissors['keyFrame'] == 1]
    Bipolar_electrocoagulation_1 = Bipolar_electrocoagulation[Bipolar_electrocoagulation['keyFrame'] == 1]

    # Get the coordinates corresponding to keyframes
    key_x = []
    key_y = []

    key_data0 = Ligation_clip_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = scissors_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = electric_hook_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = Right_angle_separating_pliers_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = Separating_pliers_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = Bipolar_electrocoagulation_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_data0 = Intestinal_forceps_1.iloc[:, [0, 1]]
    key_x0 = key_data0.values[:, 0]
    key_y0 = key_data0.values[:, 1]
    key_x.append(key_x0)
    key_y.append(key_y0)

    key_v_x = []
    key_v_y = []

    # Get the velocity corresponding to keyframes
    key_v_data0 = ac_Ligation_clip_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_scissors_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_electric_hook_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_Right_angle_separating_pliers_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_Separating_pliers_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_Bipolar_electrocoagulation_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    key_v_data0 = ac_Intestinal_forceps_1.iloc[:, [0, 1]]
    key_v_x0 = key_v_data0.values[:, 0]
    key_v_y0 = key_v_data0.values[:, 1]
    key_v_x.append(key_v_x0)
    key_v_y.append(key_v_y0)

    # area corresponding to keyframes
    key_area = []

    key_area0 = Ligation_clip_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = scissors_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = electric_hook_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = Right_angle_separating_pliers_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = Separating_pliers_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = Bipolar_electrocoagulation_1.iloc[:, [2]]
    key_area.append(key_area0)

    key_area0 = Intestinal_forceps_1.iloc[:, [2]]
    key_area.append(key_area0)

    return sum_frame, key_x, key_y, key_v_x, key_v_y, key_area



# ------------------------------------------- Get information and create a model ---------------------------------------------
tool_names = ['Ligation_clip', 'scissors', 'electric_hook', 'Right_angle_separating_pliers', 'Separating_pliers',
              'Bipolar electrocoagulation', 'Intestinal_forceps']
sum_data, key_x, key_y, key_v_x, key_v_y, key_area = get_all_list()


# Set the list of models used by different instruments
hmms = []
kenal_lim = []
kenal_v_lim_v = []
area_min_max = []
i_list = 0

# Get position model, velocity model, area threshold
while i_list < len(tool_names):
    area_min_max.append(area_lim(key_area[i_list]))
    kenal_lim.append(key_kde(key_x[i_list], key_y[i_list]))
    # kenal_lim[][0] is model，kenal_lim[][1] is lim
    kenal_v_lim_v.append(key_kde(key_v_x[i_list], key_v_y[i_list]))
    hmms.append(hmm_model(sum_data[i_list]))
    i_list += 1


# ----------------------------prediction-------------------------------------
long_data = pd.read_csv('domin_tool.csv')
long_data = np.array(long_data)

i_row = 0
while i_row + 30 < len(long_data):

    print(i_row)

    name = long_data[i_row][0]
    i_each = 1
    detect = 0
    while i_each < 30:
        long_data[i_row + i_each][6] = 0
        if long_data[i_row + i_each][0] != name or math.isnan(long_data[i_row + i_each][1]):
            # stack frames into the pool to wait for decision
            i_row += i_each - 1
            break
        if i_each == 29:
            detect = 1

        i_each += 1

    if detect == 1:
        tool = -1
        if str(name).startswith('L'):
            tool = 0
        elif str(name).startswith('sc'):
            tool = 1
        elif str(name).startswith('e'):
            tool = 2
        elif str(name).startswith('R'):
            tool = 3
        elif str(name).startswith('Sp'):
            tool = 4
        elif str(name).startswith('B'):
            tool = 5
        elif str(name).startswith('I'):
            tool = 6

        last_x_y = [long_data[i_row][1], long_data[i_row][2]]
        i_judge = i_row

        while i_judge < i_row + 30:
            # position density
            if kenal_lim[tool][0].evaluate([long_data[i_judge][1], long_data[i_judge][2]]) > kenal_lim[tool][1]:
                # area
                if area_min_max[tool][0] <= long_data[i_judge][3] <= area_min_max[tool][1]:
                    # velocity
                    if kenal_v_lim_v[tool][0].evaluate([long_data[i_judge][1] - last_x_y[0], long_data[i_judge][2] - last_x_y[1]])[0] > kenal_v_lim_v[tool][1]:
                        # tip state
                        # q_44 += 1
                        if tool == 2 or tool == 5:
                            if long_data[i_judge][5] > 0.45:
                                long_data[i_judge][6] = 1


                        elif i_judge == i_row + 29:
                            # tip state
                            long_data_tip = long_data[i_row:i_row + 29, 4]
                            tip_means = numpy.array(
                                hmms[tool].predict(long_data_tip.reshape(-1, 1).astype(int)[:])).mean()
                            if tip_means > 0.5:
                                for every_set in range(30):
                                    long_data[i_row + every_set][6] = 1

            # record prior data
            last_x_y = [long_data[i_judge][1], long_data[i_judge][2]]
            i_judge += 1

        i_row += 30

    i_row += 1


# ----------------------------write to csv-------------------------------------
title = ["class", "center_x", "center_y", "area", "num_tip",
         "lightness", "keyFrame"]

with open("keyframe.csv", "w+", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(title)

    for every_row in long_data:
        writer.writerow(every_row)

csv_file.close()
