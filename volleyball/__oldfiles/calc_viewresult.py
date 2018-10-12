import numpy as np


def calc_hsv_asym(correspondent_point, calculate_slope, input_time):

    score = [0]
    corr = correspondent_point["corr_point"]
    if not calculate_slope:
        """
        pointless
        x = correspondent_point["corr_point"][i][0] - correspondent_point["corr_point"][0][0]
        y = correspondent_point["corr_point"][i][1] - correspondent_point["corr_point"][0][1]

        if x != 0:
            slope.append(float(y) / x)
        else:
            slope.append(np.nan)
        """
        # for not calc
        ini_y = corr[0][1]
        if input_time is None: # for ref

            for i in range(1, len(corr)):
                diff = corr[i][1] - (ini_y + i)
                score.append(diff)
        else: # for input
            score = [np.nan for i in range(ini_y - 1)]
            score.append(0)
            """
            y_index = 1
            corr_y_list = np.array(corr).T[1]
            seaq_bool = False
             
            while True:

                overlaps = np.where(corr_y_list == ini_y + y_index)[0]
                if len(overlaps) == 0: # skip
                    if seaq_bool:
                        break
                    score.append((corr[y_index][0] + corr[y_index - 1][0]) / 2.0 - (y_index + 0.5))
                    print(corr[y_index][0])
                    print(corr[y_index - 1][0])
                    seaq_bool = True

                else:
                    score.append(corr[overlaps[-1]][0] - y_index)
                    seaq_bool = False

                y_index += 1
                if y_index == len(corr):
                    break
            while input_time > len(score):
                score.append(np.nan)
            """
            y_time = {}
            for i in range(len(corr)):
                y_time[corr[i][1]] = i
            y_index = 1
            y_time_list = y_time.keys()
            while y_index < len(corr):

                if y_time_list.count(ini_y + y_index) == 0:  # skip
                    try:
                        score.append((corr[y_time[ini_y + y_index + 1]][0] + corr[y_time[ini_y + y_index - 1]][
                            0]) / 2.0 - y_index)
                    except KeyError:

                        break

                    y_index += 1
                else:
                    score.append(float(corr[y_time[ini_y + y_index]][0]) - y_index)
                    y_index += y_time_list.count(ini_y + y_index)

            while input_time > len(score):
                score.append(np.nan)

        score = np.array(score)


        hsvlist = np.zeros((len(score), 3))
        max = np.nanmax(score)
        min = np.nanmin(score)
        for i in range(len(score)):
            if np.isnan(score[i]):
                hsvlist[i, 0] = 1 / 3.0  # h
                hsvlist[i, 1] = 1  # s
                hsvlist[i, 2] = 1  # v
            elif score[i] < 0:
                hsvlist[i, 0] = 2 / 3.0  # h
                hsvlist[i, 1] = 1  # s
                if min != 0:
                    hsvlist[i, 2] = score[i] / float(min)  # v
                else:
                    hsvlist[i, 2] = 0  # v
            elif score[i] == 0:
                hsvlist[i, 0] = 0  # h
                hsvlist[i, 1] = 0  # s
                hsvlist[i, 2] = 0  # v
            else:
                hsvlist[i, 0] = 0  # h
                hsvlist[i, 1] = 1  # s
                if max != 0:
                    hsvlist[i, 2] = score[i] / float(max)  # v
                else:
                    hsvlist[i, 2] = 0  # v
        return hsvlist

    else:  # slope(5 points), h = 1
        if input_time is None: # for ref
            for i in range(1, len(corr)):
                if i == 1 or i == len(corr) - 1 or i == len(corr) - 2:
                    score.append(0)
                    continue

                h = corr[i][0] - corr[i - 1][0]

                if h != 0:
                    score.append(
                        float(corr[i - 2][1] - 8 * corr[i - 1][1] + 8 * corr[i + 1][1] - corr[i][1]) / (12 * h))
                else:
                    score.append(np.nan)
        else: # for input
            ini_y = corr[0][1]
            score = [np.nan for i in range(ini_y - 1)]
            score.append(0)

            y_time = {}
            for i in range(len(corr)):
                y_time[corr[i][1]] = i
            y_index = 1
            y_time_list = y_time.keys()
            x_list = [0]
            while y_index < len(corr):

                if y_time_list.count(ini_y + y_index) == 0:  # skip
                    try:
                        x_list.append((corr[y_time[ini_y + y_index + 1]][0] + corr[y_time[ini_y + y_index - 1]][0]) / 2.0)
                    except KeyError:
                        break
                    y_index += 1
                else:
                    x_list.append(float(corr[y_time[ini_y + y_index]][0]))
                    y_index += y_time_list.count(ini_y + y_index)

            for i in range(1, len(x_list)):
                if i == 1 or i == len(x_list) - 1 or i == len(x_list) - 2:
                    score.append(0)
                    continue

                score.append(float(x_list[i - 2] - 8 * x_list[i - 1] + 8 * x_list[i + 1] - x_list[i]) / 12 * 1.0)
            while input_time > len(score):
                score.append(np.nan)


        score = np.array(score) - 1
        hsvlist = np.zeros((len(score), 3))
        # 0,1,2
        max = np.nanmax(score)
        min = np.nanmin(score)
        for i in range(len(score)):
            if np.isnan(score[i]):
                hsvlist[i, 0] = 1 / 3.0  # h
                hsvlist[i, 1] = 1  # s
                hsvlist[i, 2] = 1  # v
            elif score[i] < 0:
                hsvlist[i, 0] = 2 / 3.0  # h
                hsvlist[i, 1] = 1  # s
                hsvlist[i, 2] = score[i] / min  # v
            elif score[i] == 0:
                hsvlist[i, 0] = 0  # h
                hsvlist[i, 1] = 0  # s
                hsvlist[i, 2] = 0  # v
            else:
                hsvlist[i, 0] = 0  # h
                hsvlist[i, 1] = 1  # s
                hsvlist[i, 2] = score[i] / max  # v

        return hsvlist