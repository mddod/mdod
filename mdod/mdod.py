# -*-coding: utf-8- -*-
#MDOD, Multi-Dimensional data Outlier Detection
# Author: Z Shen<626456708@qq.com>
# License: BSD 3-Clause License

import numpy as np

def md(dets0,nd,sn): 
    VCS_list = []
    VCSResult_list = []
    i=0
    VCS_list.clear()
    for line0 in dets0:
        VCSTotal = 0
        DenominatorLeft = 0
        DenominatorRight = 0
        Numerator1 = 0
        Numerator2 = 0
        NumeratorPlus = 0
        DenominatorSum = 0
        NumeratorSum = 0
        VCSResult = 0
        j = 0
        VCSResult_list.clear()
        line0size=np.array(line0)
        for line1 in dets0:
            VCSResult = 0
            DenominatorLeft = 0
            DenominatorRight = 0
            Numerator1 = 0
            Numerator2 = 0
            NumeratorPlus = 0
            DenominatorSum = 0
            NumeratorSum = 0
            VCSResult = 0
            for k in range(0, (line0size.size), 1):
                DenominatorLeft = float(DenominatorLeft) + (float(str(line0[k]))-float(str(line0[k])))**2
            DenominatorLeft = float(DenominatorLeft) + (float(str(nd))-float(str('0')))**2
            DenominatorLeft = (float(DenominatorLeft)**0.5)
            for k in range(0, (line0size.size), 1):
                DenominatorRight = float(DenominatorRight) + (float(str(line0[k]))-float(str(line1[k])))**2
            DenominatorRight = float(DenominatorRight) + (float(str(nd))-float(str('0')))**2
            DenominatorRight = (float(DenominatorRight)**0.5)
            for k in range(0, (line0size.size), 1):
                NumeratorSum = NumeratorSum + (float(((float(str(line0[k]))-float(str(line0[k])))**2)**0.5) * float(((float(str(line0[k]))-float(str(line1[k])))**2)**0.5))
            NumeratorPlus = (float(((float(str(nd))-float(str('0')))**2)**0.5) * float(((float(str(nd))-float(str('0')))**2)**0.5))   
            DenominatorSum = float(DenominatorLeft) * float(DenominatorRight)
            NumeratorSum = NumeratorSum + NumeratorPlus
            VCSResult = float(NumeratorSum)/float(DenominatorSum)
            VCSResult_list.append(VCSResult)
        VCSResult_list.sort(key=None, reverse=True)
        for j in range(1, sn):
            VCSTotal = VCSTotal + float(str(VCSResult_list[j]))
        VCS_list.append([VCSTotal, str(line0), str(i)])
        i = i + 1
    return VCS_list



