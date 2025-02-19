# -*-coding: utf-8- -*-
#MDOD, Multi-Dimensional data Outlier Detection
# Author: Z Shen<626456708@qq.com>
# License: BSD 3-Clause License

import numpy as np

def md(dets0, nd, sn):
    VCS_list = []
    i = 0

    for line0 in dets0:
        VCSResult_list = []
        line0_arr = np.array(line0, dtype=float)  

        for j, line1 in enumerate(dets0):
            if j == i:  
                continue

            line1_arr = np.array(line1, dtype=float)

            DenominatorLeft = np.sqrt(np.sum((line0_arr - line0_arr) ** 2) + (nd - 0) ** 2)
            DenominatorRight = np.sqrt(np.sum((line0_arr - line1_arr) ** 2) + (nd - 0) ** 2)
            DenominatorSum = DenominatorLeft * DenominatorRight

            NumeratorSum = np.sum(np.sqrt((line0_arr - line0_arr) ** 2) * np.sqrt((line0_arr - line1_arr) ** 2))
            NumeratorPlus = np.sqrt((nd - 0) ** 2) * np.sqrt((nd - 0) ** 2)
            NumeratorSum += NumeratorPlus

            VCSResult = 0 if DenominatorSum == 0 else NumeratorSum / DenominatorSum
            VCSResult_list.append(VCSResult)

        VCSResult_list.sort(reverse=True)
        VCSTotal = sum(VCSResult_list[:min(sn, len(VCSResult_list))]) 

        VCS_list.append([VCSTotal, line0.tolist(), i])  
        i += 1

    return VCS_list
