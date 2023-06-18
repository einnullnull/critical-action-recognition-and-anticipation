import pandas as pd
import numpy as np
import csv
import glob

# # keyframe time, video time, file name, number of frames


long_data = pd.read_csv('detect_result.csv', low_memory=False)
long_data = np.array(long_data)
frame = []
# # Follow the order of primary instrument

# csv to onehot
for every_row in long_data:
    layer = []
    tool_class = [0, 0, 0, 0, 0, 0, 0]
    tool_postion_x = [None, None, None, None, None, None, None]
    tool_postion_y = [None, None, None, None, None, None, None]
    tool_postion_area = [None, None, None, None, None, None, None]
    tool_tip = [0, 0, 0, 0, 0, 0, 0]
    tool_angle = [None, None, None, None, None, None, None]
    lightness = every_row[0]
    row_index = 6
    while row_index < len(every_row):
        if str(every_row[row_index]) == "nan":
            break
        elif str(every_row[row_index]).startswith("Lig"):
            i = 0
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])

        elif str(every_row[row_index]).startswith("sci"):
            i = 1
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])
        elif str(every_row[row_index]).startswith("el"):
            i = 2
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])
        elif str(every_row[row_index]).startswith("Sep"):
            i = 3
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])
        elif str(every_row[row_index]).startswith("Righ"):
            i = 4
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])
        elif str(every_row[row_index]).startswith("Bipolar"):
            i = 5
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])
        elif str(every_row[row_index]).startswith("Intestinal"):
            i = 6
            tool_class[i] = 1
            tool_postion_x[i] = float(every_row[row_index - 5])
            tool_postion_y[i] = float(every_row[row_index - 4])
            tool_postion_area[i] = float(every_row[row_index - 3])
            tool_tip[i] = int(every_row[row_index - 2])
            if str(every_row[row_index - 1]) != 'None':
                tool_angle[i] = float(every_row[row_index - 1])

        row_index += 6

    layer = [tool_class, tool_postion_x, tool_postion_y, tool_postion_area, tool_tip, tool_angle, lightness]
    frame.append(layer)


name_tools = ['Ligation clip', 'scissors', 'electric hook', 'Separating pliers', 'Right angle separating pliers',
              'Bipolar electrocoagulation', 'Intestinal forceps']

title = ["class", "center_x", "center_y", "area", "num_tip", "angle",
         "lightness", "keyFrame"]
index = 0

# Only the primary instrument information is retained
with open("domin_tool.csv", "w+", newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(title)

    for every_row in frame:
        row = [None, None, None, None, None, None,
               0.0, 0]
        row[6] = every_row[6]
        i_for = 0

        i = 0

        while i < 7:
            if every_row[0][i] > 0:

                row[0] = name_tools[i]  # class
                row[1] = every_row[1][i]  # cx
                row[2] = every_row[2][i]  # cy
                row[3] = every_row[3][i]  # area
                row[4] = every_row[4][i]  # num_tip
                row[5] = every_row[5][i]  # angle

                break

            i += 1

        writer.writerow(row)
        index += 1

csv_file.close()
