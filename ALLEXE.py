import os
from ALLidentifcation import ALL_START_NFsB
from ALLidentifcation import initialize_parameters
from ALLidentifcation import initialize_parameters_all

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
# Default values, or customizable for modification
thickness, nz, dstep1= initialize_parameters_all()
ftstep2n, dtstep2n, dstep2n= initialize_parameters("n")
ftstep2fs, dtstep2fs, dstep2fs= initialize_parameters("fs")
ftstep2b, dtstep2b, dstep2b= initialize_parameters("b")


# Set base parameters
material = "ST"
inpname_tensile = "ST_tensile"
inpname_arcan = "ST_arcan"
UMATname = "umat_SMFS_model_V5"
cpus = 2
jobnumber = 3
# Set the range for the hardening exponent N
ln = 0
un = 0.2
# Set the range for the fracture strain fs
lfs = 0
ufs = 1.5
# Set the range for the material dependant parameters B
lb = 1
ub = 4
# ---------------------Starting Calculation---------------------
import os
import csv
import sys
import time
import mpmath
import pickle
import shutil
import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from itertools import accumulate
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from itertools import zip_longest

from ALLidentifcation import DataRF2
from ALLidentifcation import DataU2
def calculate_error(data, exp, Excurve):
    def Fuction_Excurve(U, exp, expi):
        exRF22 = np.array([expi(u) if u < exp[-1][0] else 0 for u in U])
        return exRF22

    dataRF2 = [row[1] for row in data]
    dataU2 = [row[0] for row in data]
    if dataU2[-1] < exp[-1][0]:
        dataU220, dataRF220 = zip(*[(u, rf) for u, rf in zip(dataU2, dataRF2) if exp[0][0] < u < exp[-1][0]])
        dataU22 = list(dataU220) + [dataU220[-1] + i * (exp[-1][0] - dataU220[-1]) / 10 for i in range(1, 11)]
        dataRF22 = list(dataRF220) + [0] * 10
        wlist00 = [dataU22[0] - exp[0][0]] + [u2_next - u2_prev for u2_prev, u2_next in
                                              zip(dataU22[:-1], dataU22[1:])] + [exp[-1][0] - dataU22[-1]]
        wlist0 = [w / (exp[-1][0] - exp[0][0]) for w in wlist00]
        wlist = [(w_next + w_prev) / 2 for w_prev, w_next in zip(wlist0, wlist0[1:])]
        interpolated_data = Fuction_Excurve(dataU22, exp, Excurve)
        error0 = np.sqrt((dataRF22 - interpolated_data) ** 2)
        error = sum([e0 * w for e0, w in zip(error0, wlist)])
    else:
        dataU221, dataRF221 = zip(*[(u, rf) for u, rf in zip(dataU2, dataRF2) if exp[0][0] < u < exp[-1][0]])
        wlist001 = [dataU221[0] - exp[0][0]] + [u2_next - u2_prev for u2_prev, u2_next in
                                                zip(dataU221[:-1], dataU221[1:])] + [exp[-1][0] - dataU221[-1]]
        wlist01 = [w / (dataU2[-1] - exp[0][0]) for w in wlist001]
        wlist1 = [(w_next + w_prev) / 2 for w_prev, w_next in zip(wlist01, wlist01[1:])]

        dataU222, dataRF222 = zip(*[(u, rf) for u, rf in zip(dataU2, dataRF2) if u > exp[-1][0]])
        wlist002 = [dataU222[0] - exp[-1][0]] + [u2_next - u2_prev for u2_prev, u2_next in
                                                 zip(dataU222[:-1], dataU222[1:])] + [0]
        wlist02 = [w / (dataU2[-1] - exp[0][0]) for w in wlist002]
        wlist2 = [(w_next + w_prev) / 2 for w_prev, w_next in zip(wlist02, wlist02[1:])]

        dataU22, dataRF22 = list(dataU221) + list(dataU222), list(dataRF221) + list(dataRF222)
        wlist = list(wlist1) + list(wlist2)
        interpolated_data = Fuction_Excurve(dataU22, exp, Excurve)
        error0 = np.sqrt((dataRF22 - interpolated_data) ** 2)
        error = sum([e0 * w for e0, w in zip(error0, wlist)])
    return error,dataU22, dataRF22, interpolated_data

# with open('Data_fs.pkl', 'rb') as file:
#     parameters_instance = pickle.load(file)
#
# RFarea = parameters_instance.RFarea
# d = parameters_instance.d
# test1 = parameters_instance.test1
# Excurve = parameters_instance.Excurve
# RFnode = parameters_instance.RFnode
#
# Case_dat_name="T02_fs_1.11397.dat"
# datdata = open(Case_dat_name, "r", encoding='utf-8').readlines()
# dataRF2 = np.array(DataRF2(datdata, RFnode), dtype=float) / RFarea
# dataU2 = (1 / d) * np.array(DataU2(datdata), dtype=float)
# data = np.column_stack((dataU2, dataRF2))
# error, dataU22, dataRF22, interpolated_data = calculate_error(data, test1, Excurve)

# with open("output_filename.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([dataU22, dataRF22, interpolated_data])

# print(error)






def ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter):
    def moveR(data, i):
        result = []
        for n in range(len(data)):
            if n <= i:
                result.append(sum(data[:2 * n + 1]) / (2 * n + 1))
            elif n >= len(data) - i - 2:
                result.append(sum(data[2 * n - len(data) + 1:]) / (2 * (len(data) - n - 1) + 1))
            elif i + 1 <= n <= len(data) - i - 3:
                result.append(sum(data[n - i:n + i + 1]) / (2 * i + 1))
        return result

    # Import SS ".csv"
    test1 = np.genfromtxt(SScsv_name, delimiter=',', skip_header=1, encoding='utf-8')

    test10 = test1                                       ### not include 0,0
    test1 = np.vstack(([0, 0], test1))
    test10 = test1                                       ### includes 0,0 for Material ST
    Excurve = interp1d(test1[:, 0], test1[:, 1])
    Mns = test1[test1[:, 1].argmax()]
    testmae = [row for row in test10 if row[1] < bline]
    testato = [row for row in test10 if row[1] > bline]
    testmae2 = moveR(testmae, averageformer)
    testato2 = moveR(testato, averagelatter)
    mtest0 = np.concatenate((testmae2, testato2))
    test00 = [item for item in mtest0 if item[0] < Mns[0]]
    true0 = [[np.log(1 + row[0]), row[1] * (1 + row[0])] for row in test00]
    true1 = [[row[0] - row[1] / EE, row[1]] for row in true0]
    trueminus1 = [[0, 0]] + true1[:-1]
    trueminus = np.array(true1) - np.array(trueminus1)
    zeroposition = np.where(trueminus[:, 0] < 0)[0][-1] + 1
    true2 = true1[zeroposition - 1:]                      ### it starts from the first point of growth
    true2 = true1[zeroposition:]                          ### the second point for Material ST
    zeropoint = true2[0][0]
    true3 = np.array([[row[0] - zeropoint, row[1]] for row in true2])
    true3 = np.array(true3)
    dydx = np.gradient(true3[:, 0]) / np.gradient(true3[:, 1])
    trueint = CubicHermiteSpline(true3[:, 1], true3[:, 0], dydx)
    truecurve = np.array([[trueint(y), y] for y in np.arange(true3[0, 1], true3[-1, 1])])
    m = truecurve[-1]
    slope = (truecurve[-1, 1] - truecurve[-2, 1]) / (truecurve[-1, 0] - truecurve[-2, 0])

    print(Mns, slope, m)

    return test1, Excurve, true3, Mns, truecurve, slope, m










bline = 300
averageformer = 250
averagelatter = 30
materialcsv_name = material + "_elastic.csv"
with open(materialcsv_name, 'r') as csvfile:
    reader = csv.reader(csvfile)
    EE = float(next(reader)[0])
    nu = float(next(reader)[0])

SScsv_name="ST_ss_ss.csv"
test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
# print(Mns, slope, m )
# with open("ss.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([test1, true3,truecurve])

SScsv_name="ST_ss.csv"
test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
# print(Mns, slope, m )

SScsv_name="ST_ss0.csv"
test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
# print(Mns, slope, m )
# with open("ss0.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([test1, true3,truecurve])