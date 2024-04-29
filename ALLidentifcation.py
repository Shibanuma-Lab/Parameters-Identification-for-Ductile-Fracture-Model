# @Software : PyCharm

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

#### Parameter Identification Section for N
class ParametersResults_N:
    def __init__(self, Common_Ninp_name, materialcsv_name, materialcsv_name2, materialParameter, Case_name,
                 RFnode, RFarea, d,
                 test1, Excurve, true3, Mns,
                 truecurve, m, slope,
                 u, l, cpus, UMATname, residualerror, identification_type):
        self.Common_Ninp_name = Common_Ninp_name
        self.materialcsv_name = materialcsv_name
        self.materialcsv_name2 = materialcsv_name2
        self.materialParameter = materialParameter
        self.Case_name = Case_name

        self.RFnode = RFnode
        self.RFarea = RFarea
        self.d = d

        self.test1 = test1
        self.Excurve = Excurve
        self.true3 = true3
        self.Mns = Mns

        self.truecurve = truecurve
        self.m = m
        self.slope = slope

        self.cpus = cpus
        self.UMATname = UMATname
        self.u = u
        self.l = l
        self.residualerror = residualerror
        self.identification_type = identification_type

def Hinpmaker(material, inpname, ftstep1, dtstep1, dstep1, frestep1, ftstep2, dtstep2, dstep2, frestep2):
    # data names
    materialcsv_name = material + "_elastic.csv"
    original_inp_name = inpname + ".inp"
    Common_Ninp_name = material + "_" + inpname + "_N.inp"

    # Import ".csv"
    with open(materialcsv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        EE = float(next(reader)[0])
        nu = float(next(reader)[0])
    # Import ".inp" data
    inp0 = open(original_inp_name, "r", encoding='utf-8').readlines()
    material_line_index = next(i for i, line in enumerate(inp0) if "*Material" in line)
    inp00 = inp0[:material_line_index + 1]

    # Set paraA, paraB and nSDV
    paraA, paraB = 10000, 0
    nSDV, SDVlist, iSDVf, odb = SDVlist_parameter()
    usermaterial = [EE, nu, paraA, paraB]
    step1 = [["*Step", "name=Step-1", "nlgeom=YES", "inc=100000"],
             ["*Static"],
             [ftstep1, 1.0, "1.e-8", dtstep1],
             ["*Boundary"],
             ["yupper", 2, 2, dstep1],
             ["ylower", 2, 2, -dstep1],
             ["*Output", "field", "frequency=" + str(frestep1)],
             ["*Node Output"],
             ["RF", "U"],
             ["*Element Output", "directions=YES"],
             odb,
             ["*Node Print", "nset=yupper"],
             ["RF2"],
             ["*Node Print", "nset=upperU"],
             ["U2"],
             ["*EL Print", "elset=center"],
             ["SDV"],
             ["*Output", "history"],
             ["*End Step"]]
    step2 = [["*Step", "name=Step-2", "nlgeom=YES", "inc=100000"],
             ["*Static"],
             [ftstep2, 1.0, "1.e-8", dtstep2],
             ["*Boundary"],
             ["yupper", 2, 2, dstep2],
             ["ylower", 2, 2, -dstep2],
             ["*Output", "field", "frequency=" + str(frestep2)],
             ["*Node Output"],
             ["RF", "U"],
             ["*Element Output", "directions=YES"],
             odb,
             ["*Node Print", "nset=yupper"],
             ["RF2"],
             ["*Node Print", "nset=upperU"],
             ["U2"],
             ["*EL Print", "elset=center"],
             ["SDV"],
             ["*Output", "history"],
             ["*End Step"]]

    # Export to a new ".inp" file
    inp10000 = inp00
    inp10001 = [["*Depvar", "delete=" + str(iSDVf)], [nSDV]]
    inp10002 = SDVlist
    inp10003 = [["*User Material", "constants=" + str(4)]] + [usermaterial] + [["*Boundary"], ["zboundary", "ZSYMM"],
                                                                               ["xpin", "XSYMM"]] + step1 + step2
    with open(Common_Ninp_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        outfile.writelines(line for line in inp10000)
        writer.writerows(inp10001)
        outfile.writelines(line + "\n" for line in inp10002)
        writer.writerows(inp10003)
    print('Finish N_inpmaker')

def ParametersProcessing(material, inpname, UMATname, u, l, RFarea, d, identification_type,
                         bline, averageformer, averagelatter, cpus, residualerror):
    # data names
    materialcsv_name = material + "_elastic.csv"
    SScsv_name = material + "_ss0.csv"
    Common_Ninp_name = material + "_" + inpname + "_N.inp"
    Case_name = inpname + "_n_"

    # Import ".csv"
    with open(materialcsv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        EE = float(next(reader)[0])
        Nu = float(next(reader)[0])
    paraA, paraB = 10000, 0
    materialParameter = [EE, Nu, paraA, paraB]

    # Get PlasticCurve Parameters
    test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
    RFnode = Getinp_RFnode(Common_Ninp_name)
    result = ParametersResults_N(Common_Ninp_name, materialcsv_name, SScsv_name, materialParameter, Case_name,
                                 RFnode, RFarea, d,
                                 test1, Excurve, true3, Mns,
                                 truecurve, m, slope,
                                 u, l, cpus, UMATname, residualerror, identification_type)
    with open('Data_N.pkl', 'wb') as f:
        pickle.dump(result, f)
    print('Finish N_ParametersProcessing')
    return result

def Tensileinpmaker(nn, parameters_instance):
    # Import and Define Parameters
    slope = parameters_instance.slope
    m = parameters_instance.m
    truecurve = parameters_instance.truecurve
    true3 = parameters_instance.true3
    Common_Ninp_name = parameters_instance.Common_Ninp_name
    materialParameter = parameters_instance.materialParameter
    Case_name = parameters_instance.Case_name
    Case_inp_name = Case_name + str(nn) + ".inp"

    # Fit PlasticCurve
    fitcurve = PlasticCurve(nn, true3, slope, m, truecurve)

    # Read and Write inp
    data0 = open(Common_Ninp_name, 'r', encoding='utf-8').readlines()
    a1 = next(i for i, line in enumerate(data0) if "*User Material" in line)
    b1 = next(i for i, line in enumerate(data0) if "*Boundary" in line)
    data1_flat = [item for sublist in zip([point[1] for point in fitcurve], [point[0] for point in fitcurve]) for item
                  in sublist]
    data1 = materialParameter + data1_flat
    data0001 = data0[:a1]
    data0002 = [(data0[a1].split(',')[0], f"constant={len(data1)}")]
    data0003 = [data1[i:i + 8] for i in range(0, len(data1), 8)]
    data0004 = data0[b1:]

    with open(Case_inp_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        outfile.writelines(line for line in data0001)
        writer.writerows(data0002)
        writer.writerows(data0003)
        outfile.writelines(line for line in data0004)

def TensileAbaqusdo(nn, parameters_instance):
    UMATname = parameters_instance.UMATname
    cpus = parameters_instance.cpus
    Case_name = parameters_instance.Case_name
    Case_bat_name = f"make_{Case_name}{nn}.bat"

    with open("make023.bat", "r") as make023_file:
        make023 = make023_file.readlines()
    make023[5] = f"ab2023 job={Case_name}{nn} user={UMATname} cpus={cpus}\n"

    with open(Case_bat_name, "w") as make023_output:
        make023_output.writelines(make023)
    time.sleep(5)
    try:
        subprocess.Popen(["cmd.exe", "/c", f"start {Case_bat_name}"], shell=True)
        time.sleep(10)
    except Exception as e:
        print("Error:", e)

def Tensileendlog(nn, parameters_instance):
    Case_name = parameters_instance.Case_name
    Case_log_name = f"{Case_name}{nn}.log"
    fin = []
    fin2 = []
    start_time = time.time()

    while not fin and not fin2:
        with open(Case_log_name, "r") as logfile:
            lines = logfile.readlines()
        fin = [i for i, line in enumerate(lines) if "COMPLETED" in line]
        fin2 = [i for i, line in enumerate(lines) if "Wrap-up" in line]
        time.sleep(10)
        current_time = time.time()
        elapsed_time = (current_time - start_time) / 60
        sys.stdout.write(f"\rCase: {Case_name}{nn}, Costed Time: {elapsed_time:.2f} minutes")  # Real-time output and overwrite the previous line
        sys.stdout.flush()
    time.sleep(10)
    print("   OVER   ", end="")

def Tensileerrormin(nn, parameters_instance):
    RFarea = parameters_instance.RFarea
    d = parameters_instance.d
    test1 = parameters_instance.test1
    Excurve = parameters_instance.Excurve
    RFnode = parameters_instance.RFnode
    Case_name = parameters_instance.Case_name
    Case_dat_name = f"{Case_name}{nn}.dat"

    datdata = open(Case_dat_name, "r", encoding='utf-8').readlines()
    dataRF2 = np.array(DataRF2(datdata, RFnode), dtype=float) / RFarea
    dataU2 = (1 / d) * np.array(DataU2(datdata), dtype=float)
    dataU22, dataRF22 = zip(*[(u, rf) for u, rf in zip(dataU2, dataRF2) if test1[0][0] < u < test1[-1][0]])
    wlist00 = [dataU22[0] - test1[0][0]] + [u2_next - u2_prev for u2_prev, u2_next in
                                            zip(dataU22[:-1], dataU22[1:])] + [test1[-1][0] - dataU22[-1]]
    wlist0 = [w / (test1[-1][0] - test1[0][0]) for w in wlist00]
    wlist = [(w_next + w_prev) / 2 for w_prev, w_next in zip(wlist0, wlist0[1:])]
    interpolated_data = Excurve(dataU22)
    error0 = np.sqrt((dataRF22 - interpolated_data) ** 2)
    error = sum([e0 * w for e0, w in zip(error0, wlist)])
    print('Error:', error)
    return error

#### Parameter Identification Section for FS
class ParametersResults_Fs:
    def __init__(self, Common_FSinp_name, materialcsv_name, materialcsv_name2, materialParameter, Case_name,
                 RFnode, RFarea, d,
                 test1, Excurve, true3, Mns,
                 truecurve, m, slope,
                 u, l, cpus, UMATname, residualerror, identification_type):
        self.Common_FSinp_name = Common_FSinp_name
        self.materialcsv_name = materialcsv_name
        self.materialcsv_name2 = materialcsv_name2
        self.materialParameter = materialParameter
        self.Case_name = Case_name

        self.RFnode = RFnode
        self.RFarea = RFarea
        self.d = d

        self.test1 = test1
        self.Excurve = Excurve
        self.true3 = true3
        self.Mns = Mns

        self.truecurve = truecurve
        self.m = m
        self.slope = slope

        self.cpus = cpus
        self.UMATname = UMATname
        self.u = u
        self.l = l
        self.residualerror = residualerror
        self.identification_type = identification_type

def FSinpmaker(material, inpname, ftstep1, dtstep1, dstep1, frestep1, ftstep2, dtstep2, dstep2, frestep2):
    # data names
    materialcsv_name = material + "_elastic.csv"
    material_plastic_csv_name = material + "_plastic.csv"
    original_inp_name = inpname + ".inp"
    outinp_name = material + "_" + inpname + "_fs.inp"

    # Import ".csv"
    with open(materialcsv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        EE = float(next(reader)[0])
        nu = float(next(reader)[0])
    hard_data = []
    with open(material_plastic_csv_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            hard_data.append([float(row[1]), float(row[0])])
    hard = [item for sublist in hard_data for item in sublist]

    # Import ".inp" data
    inp0 = open(original_inp_name, "r", encoding='utf-8').readlines()
    material_line_index = next(i for i, line in enumerate(inp0) if "*Material" in line)
    inp00 = inp0[:material_line_index + 1]

    # Set paraA, paraB and nSDV
    paraA, paraB = 0, 0
    nSDV, SDVlist, iSDVf, odb = SDVlist_parameter()


    usermaterial = [EE, nu, paraA, paraB] + hard
    usermaterial = [usermaterial[i:i + 8] for i in range(0, len(usermaterial), 8)]
    step1 = [["*Step", "name=Step-1", "nlgeom=YES", "inc=100000"],
             ["*Static"],
             [ftstep1, 1.0, "1.e-8", dtstep1],
             ["*Boundary"],
             ["yupper", 2, 2, dstep1],
             ["ylower", 2, 2, -dstep1],
             ["*Output", "field", "frequency=" + str(frestep1)],
             ["*Node Output"],
             ["RF", "U"],
             ["*Element Output", "directions=YES"],
             odb,
             ["*Node Print", "nset=yupper"],
             ["RF2"],
             ["*Node Print", "nset=upperU"],
             ["U2"],
             ["*EL Print", "elset=center"],
             ["SDV"],
             ["*Output", "history"],
             ["*End Step"]]
    step2 = [["*Step", "name=Step-2", "nlgeom=YES", "inc=100000"],
             ["*Static"],
             [ftstep2, 1.0, "1.e-8", dtstep2],
             ["*Boundary"],
             ["yupper", 2, 2, dstep2],
             ["ylower", 2, 2, -dstep2],
             ["*Output", "field", "frequency=" + str(frestep2)],
             ["*Node Output"],
             ["RF", "U"],
             ["*Element Output", "directions=YES"],
             odb,
             ["*Node Print", "nset=yupper"],
             ["RF2"],
             ["*Node Print", "nset=upperU"],
             ["U2"],
             ["*EL Print", "elset=center"],
             ["SDV"],
             ["*Output", "history"],
             ["*End Step"]]

    # Export to a new ".inp" file
    inp10000 = inp00
    inp10001 = [["*Depvar", "delete=" + str(iSDVf)], [nSDV]]
    inp10002 = SDVlist
    inp10003 = [["*User Material", "constants=" + str(len(hard) + 4)]]
    inp10004 = [["*Boundary"], ["zboundary", "ZSYMM"], ["xpin", "XSYMM"]] + step1 + step2
    with open(outinp_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        outfile.writelines(line for line in inp10000)
        writer.writerows(inp10001)
        outfile.writelines(line + "\n" for line in inp10002)
        writer.writerows(inp10003)
        writer.writerows(usermaterial)
        writer.writerows(inp10004)
    print('Finish FS_inpmaker')

def FSParametersProcessing(material, inpname, UMATname, u, l, RFarea, d, identification_type, nn,
                 bline, averageformer, averagelatter, cpus, residualerror):
    # data names
    materialcsv_name = material + "_elastic.csv"
    SScsv_name = material + "_ss.csv"
    material_plastic_csv_name = material + "_plastic.csv"
    Common_FSinp_name = material + "_" + inpname + "_fs.inp"
    original_inp_name = inpname + ".inp"
    Case_name = inpname + "_fs_"

    # Import ".csv"
    with open(materialcsv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        EE = float(next(reader)[0])
        Nu = float(next(reader)[0])
    materialParameter = [EE, Nu]

    # Get PlasticCurve Parameters & Fit PlasticCurve
    test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
    fitcurve = PlasticCurve(nn, true3, slope, m, truecurve)
    RFnode = Getinp_RFnode(original_inp_name)
    with open(material_plastic_csv_name, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(fitcurve)
    materialcsv_name3 = material + "_ss_ss.csv"
    test1 = np.genfromtxt(materialcsv_name3, delimiter=',', skip_header=1, encoding='utf-8')
    exp00 = [item[0] for item in test1][:-1]
    exp01 = [item[0] for item in test1][2:] + [test1[0][0]]
    Rotate = np.array(exp01) - np.array(exp00)
    end = np.argmax(np.where(Rotate <= 0, 1, 0))
    test1 = test1[:end]
    Excurve = interp1d(np.array(test1)[:, 0], np.array(test1)[:, 1], kind='linear')
    result = ParametersResults_Fs(Common_FSinp_name, materialcsv_name, SScsv_name, materialParameter, Case_name,
                                  RFnode, RFarea, d,
                                  test1, Excurve, true3, Mns,
                                  truecurve, m, slope,
                                  u, l, cpus, UMATname, residualerror, identification_type)
    with open('Data_fs.pkl', 'wb') as f:
        pickle.dump(result, f)
    print('Finish FSParametersProcessing')
    return result

def FSTensileinpmaker(fs, parameters_instance):
    Common_FSinp_name = parameters_instance.Common_FSinp_name
    Case_name = parameters_instance.Case_name
    Case_inp_name = Case_name + str(fs) + ".inp"

    data0 = open(Common_FSinp_name, "r", encoding='utf-8').readlines()
    a1 = next(i for i, line in enumerate(data0) if "*User Material" in line)
    line_values = data0[a1 + 1].split(',')
    line_values[2] = str(fs)
    new_line = ', '.join(line_values)
    data0[a1 + 1] = new_line
    with open(Case_inp_name, 'w', newline='') as outfile:
        outfile.writelines(line for line in data0)

def FSTensileAbaqusdo(fs, parameters_instance):
    UMATname = parameters_instance.UMATname
    cpus = parameters_instance.cpus
    Case_name = parameters_instance.Case_name
    Case_bat_name = f"make_{Case_name}{fs}.bat"

    with open("make023.bat", "r") as make023_file:
        make023 = make023_file.readlines()
    make023[5] = f"ab2023 job={Case_name}{fs} user={UMATname} cpus={cpus}\n"

    with open(Case_bat_name, "w") as make023_output:
        make023_output.writelines(make023)
    time.sleep(5)
    try:
        subprocess.Popen(["cmd.exe", "/c", f"start {Case_bat_name}"], shell=True)
        time.sleep(10)
    except Exception as e:
        print("Error:", e)

def FSTensileerrormin(fs, parameters_instance):
    RFarea = parameters_instance.RFarea
    d = parameters_instance.d
    test1 = parameters_instance.test1
    Excurve = parameters_instance.Excurve
    RFnode = parameters_instance.RFnode
    Case_name = parameters_instance.Case_name
    Case_dat_name = f"{Case_name}{fs}.dat"

    datdata = open(Case_dat_name, "r", encoding='utf-8').readlines()
    dataRF2 = np.array(DataRF2(datdata, RFnode), dtype=float) / RFarea
    dataU2 = (1 / d) * np.array(DataU2(datdata), dtype=float)
    data = np.column_stack((dataU2, dataRF2))
    error = calculate_error(data, test1, Excurve)
    print('Error:', error)
    return error

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
    return error

#### Parameter Identification Section for B and A
class ParametersResults_B:
    def __init__(self, name, Arcan_name, LDIC_name,
                 inp_name, Case_name,
                 uppern, fs, ATRI,
                 Exdata, timelist,
                 u, l, cpus, UMATname, residualerror, identification_type):
        self.name = name
        self.Arcan_name = Arcan_name
        self.LDIC_name = LDIC_name

        self.inp_name = inp_name
        self.Case_name = Case_name

        self.uppern = uppern
        self.fs = fs
        self.ATRI = ATRI

        self.Exdata = Exdata
        self.timelist = timelist

        self.cpus = cpus
        self.UMATname = UMATname
        self.u = u
        self.l = l
        self.residualerror = residualerror
        self.identification_type = identification_type

def ArcanMesh(name, material, thickness, nz, elementtype, bline, averageformer, averagelatter, nn):
    materialcsv_name = material + "_elastic.csv"
    material_plastic_csv_name = material + "_plastic.csv"
    arcan_inp_name = name + ".inp"
    outinp_name = name + "_mesh.inp"
    with open(materialcsv_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        EE = float(next(reader)[0])
        nu = float(next(reader)[0])

    if not os.path.isfile(material_plastic_csv_name):
        SScsv_name = material + "_ss.csv"
        SS0csv_name = material + "_ss0.csv"
        if os.path.isfile(SScsv_name):
            pass
        elif os.path.isfile(SS0csv_name):
            SScsv_name = SS0csv_name
        else:
            raise FileNotFoundError("Both {} and {} and {} files do not exist".format(material_plastic_csv_name,SScsv_name, SS0csv_name))
        test1, Excurve, true3, Mns, truecurve, slope, m = ParameterCurve(SScsv_name, EE, bline, averageformer, averagelatter)
        fitcurve = PlasticCurve(nn, true3, slope, m, truecurve)
        with open(material_plastic_csv_name, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(fitcurve)
    hard_data = []
    with open(material_plastic_csv_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            hard_data.append([float(row[1]), float(row[0])])
    hard = [item for sublist in hard_data for item in sublist]
    inp0 = open(arcan_inp_name, "r", encoding='utf-8').readlines()
    paraA, paraB = 0, 0
    nSDV, SDVlist, iSDVf, odb= SDVlist_parameter()

    # node and element
    tp1 = next(i for i, line in enumerate(inp0) if "*Node" in line) + 1
    tp2 = next(i for i, line in enumerate(inp0) if "*Element" in line) + 1
    tp3 = next(i for i, line in enumerate(inp0) if "*End Instance" in line) + 1
    node0 = np.array([[int(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2])]
                      for line in inp0[tp1:tp2 - 1]])
    element0 = np.array([[int(line.split(',')[0]), int(line.split(',')[1]), int(line.split(',')[2]),
                          int(line.split(',')[3]), int(line.split(',')[4])] for line in inp0[tp2:tp3 - 1]])
    node01 = np.hstack((node0, np.zeros((node0.shape[0], 1))))
    element01 = np.hstack((element0, (element0[:, 1] + len(node01)).reshape(-1, 1),
                           (element0[:, 2] + len(node01)).reshape(-1, 1), (element0[:, 3] + len(node01)).reshape(-1, 1),
                           (element0[:, 4] + len(node01)).reshape(-1, 1)))
    node = np.vstack([np.column_stack((node0[:, 0] + i * len(node0), node0[:, 1], node0[:, 2],
                                       np.full(len(node0), thickness / nz * i))) for i in range(nz + 1)])
    element = np.vstack([np.column_stack((element01[:, 0] + i * len(element01), element01[:, 1:9] + i * len(node01)))
                         for i in range(nz)])

    # Output Position
    tp = [float(item) for item in inp0[tp1 - 2].split(',')]
    zboundarynode = [node01[:, 0].flatten().tolist()[i:i + 16] for i in
                     range(0, len(node01[:, 0].flatten().tolist()), 16)]
    max_z_indices = np.where(node[:, 2] == np.max(node[:, 2]))[0]
    upperfnode = [node[max_z_indices, 0].tolist()[i:i + 16] for i in range(0, len(node[max_z_indices, 0].tolist()), 16)]
    usermaterial = [EE, nu, paraA, paraB] + hard
    usermaterial = [usermaterial[i:i + 8] for i in range(0, len(usermaterial), 8)]

    # Output
    meshinp00001 = [["*Heading"],
                    ["*Preprint", "echo=NO", "model=NO", "history=NO", "contact=NO"],
                    ["*Part", "name=Part-1"],
                    ["*End Part"],
                    ["*Assembly", "name=Assembly"],
                    ["*Instance", "name=Part-1-1", "part=Part-1"],
                    tp,
                    ["*Node"]]
    meshinp00002 = [["*Element", f"type={elementtype}"]]
    meshinp00003 = [["*Nset", "nset=nall", "generate"],
                    [1, len(node), 1],
                    ["*Elset", "elset=nall", "generate"],
                    [1, len(element), 1],
                    ["*Solid Section", "elset=nall", "material=steel"],
                    [" ", " "],
                    ["*End Instance"],
                    ["*Nset", "nset=zboundary", "instance=Part-1-1"]]
    meshinp00004 = [["*Nset", "nset=upperf", "instance=Part-1-1"]]
    meshinp00005 = [["*End Assembly"],
                    ["*Material", "name=steel"],
                    ["*Depvar"],
                    [nSDV],
                    ["*User Material", f"constants={len(hard) + 4}"]]
    meshinp00006 = [["*Boundary"],
                    ["zboundary", "ZSYMM"]]
    upperfnode = [[int(item) for item in sublist] for sublist in upperfnode]
    zboundarynode = [[int(item) for item in sublist] for sublist in zboundarynode]
    with open(outinp_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(meshinp00001)
        for row in node:
            writer.writerow(['{:.0f}'.format(item) if idx == 0 else item for idx, item in enumerate(row)])
        writer.writerows(meshinp00002)
        writer.writerows(element)
        writer.writerows(meshinp00003)
        writer.writerows(zboundarynode)
        writer.writerows(meshinp00004)
        writer.writerows(upperfnode)
        writer.writerows(meshinp00005)
        writer.writerows(usermaterial)
        writer.writerows(meshinp00006)

def ArcanStep(name, material, inpname, fs, iout, thresold, incnum, finc, mininc, maxinc, numatmpinc, elementtype, u, l, cpus, UMATname, residualerror, identification_type):
    if os.path.exists("Data_B.pkl") :
        print("Data_B.pkl already existed!    'Finish ArcanParametersProcessing")
        with open('Data_B.pkl', 'rb') as file:
            result = pickle.load(file)
        ATRI_B = result.ATRI
        if os.path.isfile("Results_fs.csv"):
            with open("Results_fs.csv", 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                data_fs = []
                error_fs = []
                for row in reader:
                    try:
                        data_fs.append(float(row[0]))
                        error_fs.append(float(row[1]))
                    except:
                        next(reader)
                        break
                Best_fs = next(reader)
                Best_result_fs = float(Best_fs[0])
                Best_error_fs = float(Best_fs[1])
                ATRI_fs = float(Best_fs[2])
                if ATRI_fs != ATRI_B:
                    result.ATRI = ATRI_fs
                    print("Change ATRI from", ATRI_B, "to Result_fs.csv's value:", ATRI_fs)
    elif os.path.exists(os.path.join(os.path.dirname(__file__), 'b', 'Data_B.pkl')):
        print("Data_B.pkl already existed!    'Finish ArcanParametersProcessing")
        with open(os.path.join(os.path.dirname(__file__), 'b', 'Data_B.pkl'), 'rb') as file:
            result = pickle.load(file)
    elif os.path.exists(os.path.join(os.path.dirname(__file__), 'ALLResults', 'Data_B.pkl')):
        print("Data_B.pkl already existed!    'Finish ArcanParametersProcessing")
        with open(os.path.join(os.path.dirname(__file__), 'ALLResults', 'Data_B.pkl'), 'rb') as file:
            result = pickle.load(file)
    else:
        def cf(s, sfacep, sc1, notcht, zerop):
            def func(x, a2, b2, c1):
                return -a2 * x[0] - b2 * x[1] - c1

            ssc = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(sfacep, s)]
            ssc = np.array(ssc)
            params, covariance = curve_fit(func, (ssc[:, 0], ssc[:, 1]), ssc[:, 2])
            ez = np.array([params[0], params[1], 1])
            ex0 = np.array(zerop) - np.array(notcht)
            ex = np.array([ex0[0], ex0[1], -(ex0[0] * ez[0] + ex0[1] * ez[1]) / ez[2]])
            ey = np.cross(ez, ex)
            ex1 = ex / np.linalg.norm(ex)
            ey1 = ey / np.linalg.norm(ey)
            ez1 = ez / np.linalg.norm(ez)
            Tc = np.linalg.inv(np.transpose([ex1, ey1, ez1]))
            sc1 = np.array(sc1)
            result = [[
                np.dot(Tc[0], ssc[i]) - sc1[i, 0],
                np.dot(Tc[1], ssc[i]) - sc1[i, 1],
                np.dot(Tc[2], ssc[i]) + params[2] - sc1[i, 2]] for i in range(len(ssc))]
            return result

        def findyyy(x, y, z, result_fuction):
            step = 0.01
            delta = 1e-6
            y1 = y
            y2 = y
            while True:
                upper_value1 = result_fuction(x, y1, z)
                upper_value2 = result_fuction(x, y2, z)
                if upper_value1 != 9999999:
                    y3 = y1 + delta
                    upper_value3 = result_fuction(x, y3, z)
                    if upper_value3 != 9999999:
                        d = (upper_value3 - upper_value1) / delta
                        val = result_fuction(x, y1, z) + d * (y - y1)
                        return y1, d, val
                if upper_value2 != 9999999:
                    y4 = y2 - delta
                    upper_value4 = result_fuction(x, y4, z)
                    if upper_value4 != 9999999:
                        d = (upper_value2 - upper_value4) / delta
                        val = result_fuction(x, y2, z) + d * (y - y2)
                        return y2, d, val
                y1 += step
                y2 -= step

        def Interpolate_ul(listtt, nstep0_np, ndatax, upperline):
            coords, values = zip(*[(item[0], item[1]) for item in listtt])
            x, y, z = zip(*coords)
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            values = np.array(values)
            result_fuction = LinearNDInterpolator((x, y, z), values, fill_value=9999999)
            result = result_fuction(nstep0_np[:, None] + 1, ndatax, upperline)
            for i, row in enumerate(result):
                for j, element in enumerate(row):
                    if element == 9999999:
                        nnn, d, a = findyyy(nstep0_np[i] + 1, ndatax[j], upperline, result_fuction)
                        result[i][j] = a
            return result.tolist()

        def Cth(x, thresold):
            ct = np.ceil(x / thresold)
            return np.array([1 / ct] * int(ct))

        def sepdis(weightttt, d1x0i, d1x00i):
            result = []
            for i in range(len(weightttt)):
                w_value = weightttt[i]
                a = [d100_i + (d10_i - d100_i) * w_value for d10_i, d100_i in zip(d1x0i, d1x00i)]
                result.append(a)
            return result

        def f(i, n1, n2, d1x, d1y, d2x, d2y, outputstep, weight, incnum, finc, mininc, maxinc):
            def fout(i, outputstep):
                if outputstep[i] == 0:
                    return "frequency=0"
                else:
                    return "frequency=10000"

            foutii = fout(i, outputstep)
            nSDV, SDVlist, iSDVf, odb = SDVlist_parameter()
            step_info = [["*Step", f"name=Step-{i + 1}", "nlgeom=YES", f"inc={incnum}"],
                         ["*Static"],
                         [finc * weight[i], weight[i], mininc * weight[i], maxinc * weight[i]],
                         ["*Controls", "PARAMETERS=TIME INCREMENTATION"],
                         ["", "", "", "", "", "", "", str(numatmpinc), "", "", "", ""],
                         ["*Boundary"]]
            d1xff = [(n, 1, 1, val) for n, val in zip(n1, d1x[i])]
            d1yff = [(n, 2, 2, val) for n, val in zip(n1, d1y[i])]
            d2xff = [(n, 1, 1, val) for n, val in zip(n2, d2x[i])]
            d2yff = [(n, 2, 2, val) for n, val in zip(n2, d2y[i])]
            step_info2 = [["*EL Print", foutii, "POSITION=CENTROIDAL"],
                          ["SDV"+ str(iSDVf)],
                          ["*Node Print", "nset=upperf", "frequency=10000"],
                          ["RF"],
                          ["*Node Print", foutii],
                          ["U"],
                          ["*Output", "field", foutii],
                          ["*Node Output"],
                          ["RF", "U"],
                          ["*Element Output", "directions=YES"],
                          odb,
                          ["*Output", "history", "variable=PRESELECT"],
                          ["*End Step"]]
            result = step_info + d1xff + d1yff + d2xff + d2yff + step_info2
            return result

        def gu(nsetu):
            Gu_string = []
            for x in range(len(nsetu)):
                result_string = ",".join(map(str, ["*Nset", f"nset=Upper{x + 1}", "instance=Part-1-1\n"]))
                Gu_string_r = ','.join(str(int(val)) for val in nsetu[x])
                Gu_string_r = [result_string, f"{Gu_string_r}\n"]
                Gu_string.extend(Gu_string_r)
            return Gu_string

        def gl(nsetl):
            Gl_string = []
            for x in range(len(nsetl)):
                result_string = ",".join(map(str, ["*Nset", f"nset=Lower{x + 1}", "instance=Part-1-1\n"]))
                Gl_string_r = ','.join(str(int(val)) for val in nsetl[x])
                Gl_string_r = [result_string, f"{Gl_string_r}\n"]
                Gl_string.extend(Gl_string_r)
            return Gl_string

        # Import data from CSV and INP files
        yB_name = name + "_yB.csv"
        Arcan_name = name + "_Arcan.csv"
        meshinp_name = name + "_mesh.inp"
        sc0_name = name + "_sc.csv"
        ud0_name = name + "_upper.csv"
        ld0_name = name + "_lower.csv"
        inp_name = name + "_A.inp"
        LDIC_name = name + "_Load.csv"
        Best_fs_dat = inpname + "_fs_" + str(fs) + ".dat"
        Case_name = name + "_A_"
        with open(yB_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            upperline = float(next(reader)[0])
            lowerline = float(next(reader)[0])
        with open(Arcan_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            picnum = float(next(reader)[0])
            dt = float(next(reader)[0])
            ftime = float(next(reader)[0])
        meshinp = open(meshinp_name, "r").readlines()
        sc0 = pd.read_csv(sc0_name)
        ud0 = pd.read_csv(ud0_name)
        ld0 = pd.read_csv(ld0_name)
        if os.path.isfile("Results_fs.csv"):
            with open("Results_fs.csv", 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                data_fs = []
                error_fs = []
                for row in reader:
                    try:
                        data_fs.append(float(row[0]))
                        error_fs.append(float(row[1]))
                    except:
                        next(reader)
                        break
                Best_fs = next(reader)
                Best_result_fs = float(Best_fs[0])
                Best_error_fs = float(Best_fs[1])
                ATRI = float(Best_fs[2])
        elif os.path.isfile(Best_fs_dat):
            datdata = open(Best_fs_dat, "r", encoding='utf-8').readlines()
            triaxiality = DAT_values(datdata, 19, 2)
            eqplas = DAT_values(datdata, 13, 5)
            ATRI = average_values(triaxiality, eqplas)
        else:
            TensileR_name = material + "_TensileR.csv"
            with open(TensileR_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                fs = float(next(reader)[0])
                ATRI = float(next(reader)[0])
        print("ATRI", ATRI)

        # Experimental data
        nstep0 = int(picnum) - 1
        t1 = meshinp.index(next(filter(lambda line: "nset=upperf" in line, meshinp))) + 1
        t2 = meshinp.index(next(filter(lambda line: "End Assembly" in line, meshinp))) + 1
        upper_line = meshinp[t1:t2 - 1]
        uppern = 0
        for i in range(len(upper_line)):
            uppernum = len(upper_line[i].split(','))
            uppern += uppernum
        Exdata0 = np.genfromtxt(LDIC_name, delimiter=',', skip_header=2, usecols=(0, 1), max_rows=nstep0,
                                encoding='utf-8')
        Exdata = interp1d(Exdata0[:, 0], Exdata0[:, 1], fill_value='extrapolate')
        timelist0 = [ftime + i * dt for i in range(nstep0 + 1)]

        # parallel moving at assemble
        nupperp = (ud0.shape[-1] - 1) // 3
        nlowerp = (ld0.shape[-1] - 1) // 3
        node_position = next(i for i, row in enumerate(meshinp) if "*Node" in row)
        element_position = next(i for i, row in enumerate(meshinp) if "*Element" in row)
        tp = [float(val) for val in meshinp[node_position - 1].split(',')]
        ndata0 = [list(map(float, row.split(','))) for row in meshinp[node_position + 1: element_position]]
        ndata = [[row[0], row[1] + tp[0], row[2] + tp[1], row[3] + tp[2]] for row in ndata0]

        # numbers of nodes in upper and lower boundary surface, nodal coordinate of boundary surface on actual length
        max_value = max(ndata, key=lambda x: x[2])[2]
        min_value = min(ndata, key=lambda x: x[2])[2]
        ndatau0 = [row for row in ndata if row[2] == max_value]
        ndatal0 = [row for row in ndata if row[2] == min_value]
        nthick = len(set(item[3] for item in ndatau0))
        ndatau = sorted(ndatau0, key=lambda x: (x[1], -x[0]))
        ndatal = sorted(ndatal0, key=lambda x: (x[1], -x[0]))
        ndataux = [item[1] for item in ndatau[::nthick]]
        ndatalx = [item[1] for item in ndatal[::nthick]]
        nsetuu = [item[0] for item in ndatau]
        nsetll = [item[0] for item in ndatal]
        nsetu = [nsetuu[i:i + nthick] for i in range(0, len(nsetuu), nthick)]
        nsetl = [nsetll[i:i + nthick] for i in range(0, len(nsetll), nthick)]

        # start coodinates
        notcht = sc0.iloc[nupperp + nlowerp, 1:4].values.tolist()
        zerop = sc0.iloc[-1, 1:4].values.tolist()
        sfacep00 = sc0.iloc[0:nupperp + nlowerp, 1:4].values.tolist()
        sfacep = np.array([[row[0] - zerop[0], row[1] - zerop[1], row[2] - zerop[2]] for row in sfacep00])
        # sface = Interpolate_ul(sfacep)
        s0000 = np.zeros_like(sfacep)
        sc1 = cf(s0000, sfacep, s0000, notcht, zerop)
        supperc = sc1[:nupperp]
        slowerc = sc1[nupperp:nupperp + nlowerp]

        # start point after coordinate transformation
        udx = np.transpose(ud0.iloc[:, 1:nupperp + 1])
        udy = np.transpose(ud0.iloc[:, nupperp + 1:2 * nupperp + 1])
        udz = np.transpose(ud0.iloc[:, 2 * nupperp + 1:3 * nupperp + 1])
        ldx = np.transpose(ld0.iloc[:, 1:nlowerp + 1])
        ldy = np.transpose(ld0.iloc[:, nlowerp + 1:2 * nlowerp + 1])
        ldz = np.transpose(ld0.iloc[:, 2 * nlowerp + 1:3 * nlowerp + 1])
        ssc0 = []
        for i in range(2, nstep0 + 2):
            u_data = np.column_stack((udx.iloc[:, i - 1], udy.iloc[:, i - 1], udz.iloc[:, i - 1]))
            l_data = np.column_stack((ldx.iloc[:, i - 1], ldy.iloc[:, i - 1], ldz.iloc[:, i - 1]))
            ssc0.append(np.concatenate((u_data, l_data), axis=0))
        ssc0 = np.array(ssc0)

        # ux, uy function of t, x, y
        ulp0 = [cf(s, sfacep, sc1, notcht, zerop) for s in ssc0]
        upperxlist0 = [[[j, supperc[i - 1][0], supperc[i - 1][1]], ulp0[j - 1][i - 1][0]] for i in range(1, nupperp + 1)
                       for j in range(1, nstep0 + 1)]
        upperylist0 = [[[j, supperc[i - 1][0], supperc[i - 1][1]], ulp0[j - 1][i - 1][1]] for i in range(1, nupperp + 1)
                       for j in range(1, nstep0 + 1)]
        lowerxlist0 = [[[j, slowerc[i - 1 - nupperp][0], slowerc[i - 1 - nupperp][1]], ulp0[j - 1][i - 1][0]] for i in
                       range(nupperp + 1, nlowerp + nupperp + 1) for j in range(1, nstep0 + 1)]
        lowerylist0 = [[[j, slowerc[i - 1 - nupperp][0], slowerc[i - 1 - nupperp][1]], ulp0[j - 1][i - 1][1]] for i in
                       range(nupperp + 1, nlowerp + nupperp + 1) for j in range(1, nstep0 + 1)]
        ndataux_np = np.array(ndataux)
        ndatalx_np = np.array(ndatalx)
        nstep0_np = np.array(range(nstep0))

        d1x0 = Interpolate_ul(upperxlist0, nstep0_np, ndataux_np, upperline)
        d1y0 = Interpolate_ul(upperylist0, nstep0_np, ndataux_np, upperline)
        d2x0 = Interpolate_ul(lowerxlist0, nstep0_np, ndatalx_np, lowerline)
        d2y0 = Interpolate_ul(lowerylist0, nstep0_np, ndatalx_np, lowerline)
        ddy0 = [np.mean(d1) - np.mean(d2) for d1, d2 in zip(d1y0, d2y0)]
        ddy = [ddy0[0]] + [ddy0[i] - ddy0[i - 1] for i in range(1, len(ddy0))]

        weight0 = [Cth(x, thresold).tolist() for x in np.abs(ddy)]
        weight = [item for sublist in weight0 if sublist for item in sublist if item is not np.nan]
        weight1 = []
        for row in weight0:
            sum_so_far = 0
            new_row = []
            for element in row:
                if element is not np.nan:
                    sum_so_far += element
                    new_row.append(sum_so_far)
                else:
                    break
            weight1.append(new_row)
        d1x00 = [[0] * len(nsetu)] + d1x0[:-1]
        d1y00 = [[0] * len(nsetu)] + d1y0[:-1]
        d2x00 = [[0] * len(nsetl)] + d2x0[:-1]
        d2y00 = [[0] * len(nsetl)] + d2y0[:-1]
        d1x = [item for w, d0, d00 in zip(weight1, d1x0, d1x00) for item in sepdis(w, d0, d00)]
        d1y = [item for w, d0, d00 in zip(weight1, d1y0, d1y00) for item in sepdis(w, d0, d00)]
        d2x = [item for w, d0, d00 in zip(weight1, d2x0, d2x00) for item in sepdis(w, d0, d00)]
        d2y = [item for w, d0, d00 in zip(weight1, d2y0, d2y00) for item in sepdis(w, d0, d00)]

        #
        timelist = [ftime] + list(map(lambda x: ftime + x * dt, accumulate(weight)))
        nstep = len(d1x)
        outputstep0 = [1 if i % iout == 0 else 0 for i in range(1, nstep0 + 1)]
        lastINC = [[0] * (len(item) - 1) + [1] for item in weight1]
        outputstep = [lastINC[i][j] * outputstep0[i] for i in range(len(lastINC)) for j in range(len(lastINC[i]))]
        outputstep[-1] = 1
        n1 = ["Upper" + str(i) for i in range(1, len(nsetu) + 1)]
        n2 = ["Lower" + str(i) for i in range(1, len(nsetl) + 1)]
        f_list = [f(i, n1, n2, d1x, d1y, d2x, d2y, outputstep, weight, incnum, finc, mininc, maxinc) for i in
                  range(nstep)]
        inpfile = [item for sublist in f_list for item in sublist]

        # Combine inpfile0 and f_list
        nSDV, SDVlist, iSDVf, odb= SDVlist_parameter()
        material_position = next(i for i, row in enumerate(meshinp) if "*User Material" in row)
        assembly_position = next(i for i, row in enumerate(meshinp) if "*End Assembly" in row)
        depvar_position = next(i for i, row in enumerate(meshinp) if "*Depvar" in row)
        meshinp[material_position + 1] = ','.join(
            meshinp[material_position + 1].split(',')[:2] + [str(0), str(0)] + meshinp[material_position + 1].split(
                ',')[4:])
        setlist = gu(nsetu) + gl(nsetl)
        inp_00001 = meshinp[: element_position]
        inp_00002 = [["*Element", f"type={elementtype}"]]
        inp_00003 = meshinp[element_position + 1: assembly_position]
        inp_00004 = setlist
        inp_00005 = meshinp[assembly_position:depvar_position]
        inp_00006 = [["*Depvar", "delete=" + str(iSDVf)], [nSDV]]
        inp_00007 = meshinp[depvar_position + 2:]
        with open(inp_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvfile.writelines(line for line in inp_00001)
            csvwriter.writerows(inp_00002)
            csvfile.writelines(line for line in inp_00003)
            csvfile.writelines(line for line in inp_00004)
            csvfile.writelines(line for line in inp_00005)
            csvwriter.writerows(inp_00006)
            csvfile.writelines(line + "\n" for line in SDVlist)
            csvfile.writelines(line for line in inp_00007)
            csvwriter.writerows(inpfile)
        result = ParametersResults_B(name, Arcan_name, LDIC_name,
                                     inp_name, Case_name,
                                     uppern, fs, ATRI,
                                     Exdata, timelist,
                                     u, l, cpus, UMATname, residualerror, identification_type)
        with open('Data_B.pkl', 'wb') as f:
            pickle.dump(result, f)
        print('Finish ArcanParametersProcessing')
    return result

def Arcaninpmaker_B(b, parameters_instance):
    fs = parameters_instance.fs
    ATRI = parameters_instance.ATRI
    inp_name = parameters_instance.inp_name
    Case_name = parameters_instance.Case_name

    A = ABCfuction(b,fs,ATRI)
    inpdata = open(inp_name, "r").readlines()
    material_position = next(i for i, row in enumerate(inpdata) if "*User Material" in row)
    inpdata[material_position + 1] = ','.join(inpdata[material_position + 1].split(',')[:2] + [str(A), str(b)] + inpdata[material_position + 1].split(',')[4:])
    A = float("{:.6g}".format(A))
    Case_inp_name = Case_name + str(A) + "_B_" + str(b) + ".inp"
    with open(Case_inp_name, 'w', newline='') as outfile:
        outfile.writelines(line for line in inpdata)

def ArcanAbaqusdo_B(b, parameters_instance):
    UMATname = parameters_instance.UMATname
    cpus = parameters_instance.cpus
    Case_name = parameters_instance.Case_name
    fs = parameters_instance.fs
    ATRI = parameters_instance.ATRI

    A = ABCfuction(b,fs,ATRI)
    A = float("{:.6g}".format(A))
    Case_bat_name = f"make_{Case_name}{A}_B_{b}.bat"
    with open("make023.bat", "r") as make023_file:
        make023 = make023_file.readlines()
    make023[5] = f"ab2023 job={Case_name}{A}_B_{b} user={UMATname} cpus={cpus}\n"
    with open(Case_bat_name, "w") as make023_output:
        make023_output.writelines(make023)
    time.sleep(5)
    try:
        subprocess.Popen(["cmd.exe", "/c", f"start {Case_bat_name}"], shell=True)
        time.sleep(10)
    except Exception as e:
        print("Error:", e)

def Arcanendlog(b, parameters_instance):
    Case_name = parameters_instance.Case_name
    fs = parameters_instance.fs
    ATRI = parameters_instance.ATRI
    A = ABCfuction(b,fs,ATRI)
    A = float("{:.6g}".format(A))
    Case_log_name = f"{Case_name}{A}_B_{b}.log"
    fin = []
    fin2 = []
    start_time = time.time()

    while not fin and not fin2:
        with open(Case_log_name, "r") as logfile:
            lines = logfile.readlines()
        fin = [i for i, line in enumerate(lines) if "COMPLETED" in line]
        fin2 = [i for i, line in enumerate(lines) if "Wrap-up" in line]
        time.sleep(10)
        current_time = time.time()
        elapsed_time = (current_time - start_time) / 60
        sys.stdout.write(f"\rCase: {Case_name}{A}_B_{b}, Costed Time: {elapsed_time:.2f} minutes")  # Real-time output and overwrite the previous line
        sys.stdout.flush()
    time.sleep(10)
    print("   OVER   ", end="")

def Arcanerrormin_B(b, parameters_instance):
    uppern = parameters_instance.uppern
    Exdata = parameters_instance.Exdata
    timelist = parameters_instance.timelist
    fs = parameters_instance.fs
    ATRI = parameters_instance.ATRI
    A = ABCfuction(b,fs,ATRI)
    A = float("{:.6g}".format(A))
    Case_name = parameters_instance.Case_name
    Case_dat_name = f"{Case_name}{A}_B_{b}.dat"

    datdata = open(Case_dat_name, "r", encoding='utf-8').readlines()
    xRF, yRF = np.array(DataRF1(datdata, uppern), dtype=float)
    RFdatas = np.sqrt(np.array(xRF) ** 2 + np.array(yRF) ** 2)

    wlist0 = [timelist[i] - timelist[i - 1] for i in range(2, len(timelist))]
    wlist = np.append(wlist0, wlist0[-1]) / (timelist[-1] - timelist[1])
    Exdata2 = Exdata(timelist[1:])
    RFdatas = np.reshape(RFdatas, (-1, 1))            #### Reshaping and aligning the data
    Exdata2 = np.resize(Exdata2, RFdatas.shape)                #### Reshaping and aligning the data
    error0 = (np.array(Exdata2) - np.array(RFdatas)) ** 2
    error = np.sqrt(np.sum([e0 * w for e0, w in zip(error0, wlist)]))
    print('Error:', error)

    output_filename = f"{Case_name}{A}_B_{b}_Load.csv"
    header = ["Ave.Error", "Time", "Stroke", "Experiment", "FEM"]
    data = [[error] + list(row) for row in zip(timelist[1:],
                                               [(t - timelist[1]) / 30 for t in timelist[1:]],
                                               Exdata2,
                                               RFdatas)]
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    return error

# # -------------------------common part for Tensile Test and Arcan Tests--------------------------------------------------
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
    return test1, Excurve, true3, Mns, truecurve, slope, m

def PlasticCurve(nn, true3, slope, m, truecurve):
    def cut(x):
        ac = 10000000
        x1 = x * ac
        x2 = round(x1)
        x3 = x2 / ac
        return round(x3, 10)

    def nf(n0):
        mpmath.dps = 15
        n1 = cut(n0)
        bfit = sp.symbols('bfit')
        afit = sp.symbols('afit')
        equations1 = [slope - (m[0] - bfit) ** (n1 - 1) * n1 * m[1] / (m[0] - bfit) ** n1]
        solve1 = sp.solve(equations1, bfit, domain=sp.Reals)
        b2 = mpmath.mpmathify(solve1[bfit])
        equations2 = [slope - n1 * afit * (m[0] - b2) ** (n1 - 1)]
        solve2 = sp.solve(equations2, afit, domain=sp.Reals)
        a2 = mpmath.mpmathify(solve2[afit])
        return n1, a2, b2

    def nzi(x, a2, b2, nn):
        import mpmath
        mpmath.mp.dps = 15
        x = mpmath.mpf(x)
        a2 = mpmath.mpf(a2)
        b2 = mpmath.mpf(b2)
        nn = mpmath.mpf(nn)
        result = a2 * (x - b2) ** nn
        return float(result)

    n1, a2, b2 = nf(nn)
    nzicurve = [[x, nzi(x, a2, b2, nn)] for x in np.arange(true3[-1][0], 10, 0.002)]
    fitcurve = np.concatenate((truecurve, nzicurve), axis=0)
    return fitcurve

#### Common part for Tensile Test
def Getinp_RFnode(inp_name):
    # Get RFnode information
    inp0 = open(inp_name, 'r', encoding='utf-8').readlines()
    t1 = next(i for i, line in enumerate(inp0) if "nset=yupper" in line) + 1
    t2 = next(i for i, line in enumerate(inp0) if "elset=yupper" in line)
    RFnode_line = inp0[t1:t2]
    sumrodenum = 0
    for i in range(len(RFnode_line)):
        rodenum = len(RFnode_line[i].split(','))
        sumrodenum += rodenum
    RFnode = sumrodenum
    return RFnode

def DataU2(data):
    u2_keyword = "NODE FOOT-  U2"
    a = [i for i, line in enumerate(data) if u2_keyword in line]
    b = list(range(1, len(a) + 1))
    resultU2_all = [data[a[i] + 3].split()[1] for i in range(len(b))]
    resultU2 = []
    for value in resultU2_all:
        try:
            num = float(value)
            resultU2.append(num)
        except ValueError:
            pass
    return resultU2

def DataRF2(data, num):
    rf2_keyword = "NODE FOOT-  RF2"
    a = [i for i, line in enumerate(data) if rf2_keyword in line]
    a = [i + 3 for i in a]
    b = [i + num - 1 for i in a]
    resultRF2 = []
    for i in range(len(a)):
        sumdata = [float(line.split()[1]) for line in data[a[i]:b[i] + 1]]
        result = sum(sumdata)
        resultRF2.append(result)
    resultRF2 = np.array(resultRF2)[~np.isnan(resultRF2)]
    return resultRF2

def From_experiment(identification_type, material):
    def read_csv(filename):
        data = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            data_lines = csvfile.readlines()
            for row in data_lines:
                values = [float(value) for value in row.strip().lstrip('\ufeff').split(',')]
                data.append(values)
        return data

    if identification_type == 'n':
        csv_name= material + "_ss0.csv"
        test1 = np.genfromtxt(csv_name, delimiter=',', skip_header=1, encoding='utf-8')
        test1 = np.vstack(([0, 0], test1))
        Excurve = interp1d(test1[:, 0], test1[:, 1])
    elif identification_type == 'fs':
        csv_name = material + "_ss_ss.csv"
        test1 = read_csv(csv_name)
        exp00 = [item[0] for item in test1][:-1]
        exp01 = [item[0] for item in test1][2:] + [test1[0][0]]
        Rotate = np.array(exp01) - np.array(exp00)
        end = np.argmax(np.where(Rotate <= 0, 1, 0))
        test1 = test1[:end]
        Excurve = interp1d(np.array(test1)[:, 0], np.array(test1)[:, 1], kind='linear')

    exp = test1
    expi = Excurve
    return exp, expi

def alldatafind(exp, expi, parameters_instance, identification_type):
    TypeName = "_" + identification_type
    files = [file for file in os.listdir() if TypeName in file and file.endswith('.dat')]
    error = []
    Nvalue = []
    RFnode = parameters_instance.RFnode
    RFarea = parameters_instance.RFarea
    d = parameters_instance.d
    for f in files:
        datdata = open(f, "r", encoding='utf-8').readlines()
        dataRF2 = np.array(DataRF2(datdata, RFnode), dtype=float) / RFarea
        dataU2 = (1 / d) * np.array(DataU2(datdata), dtype=float)
        if identification_type == 'n':
            dataU22, dataRF22 = zip(*[(u, rf) for u, rf in zip(dataU2, dataRF2) if exp[0][0] < u < exp[-1][0]])
            wlist00 = [dataU22[0] - exp[0][0]] + [u2_next - u2_prev for u2_prev, u2_next in
                                                  zip(dataU22[:-1], dataU22[1:])] + [exp[-1][0] - dataU22[-1]]
            wlist0 = [w / (exp[-1][0] - exp[0][0]) for w in wlist00]
            wlist = [(w_next + w_prev) / 2 for w_prev, w_next in zip(wlist0, wlist0[1:])]
            interpolated_data = expi(dataU22)
            error0 = np.sqrt((dataRF22 - interpolated_data) ** 2)
            error_value = sum([e0 * w for e0, w in zip(error0, wlist)])
            Nvalue_value = os.path.splitext(f)[0].split('_')[-1]
            print("N:", Nvalue_value, "Error:", error_value)
        elif identification_type == 'fs':
            data2 = np.column_stack((dataU2, dataRF2))
            data = data2
            error_value = calculate_error(data, exp, expi)
            Nvalue_value = os.path.splitext(f)[0].split('_')[-1]
            print("fs:", Nvalue_value, "Error:", error_value)
        error.append(float(error_value))
        Nvalue.append(float(Nvalue_value))
    return error, Nvalue

#### Common part for Arcan Test
def DataRF1(data, num):
    rf1_keyword = "RF1"
    RF1position = [i for i, line in enumerate(data) if rf1_keyword in line]
    a = [i + 3 for i in RF1position]
    b = [i + num - 1 for i in a]
    xRF = []
    yRF = []
    for i in range(len(a)):
        xRF_value = [float(line.split()[1]) * 2 / 1000 for line in data[a[i]:b[i]+1]]
        yRF_value = [float(line.split()[2]) * 2 / 1000 for line in data[a[i]:b[i] + 1]]
        xRF_result = sum(xRF_value)
        yRF_result = sum(yRF_value)
        xRF.append(xRF_result)
        yRF.append(yRF_result)
    xRF = np.array(xRF)[~np.isnan(xRF)]
    yRF = np.array(yRF)[~np.isnan(yRF)]
    return xRF, yRF

def allArcandatafind(parameters_instance):
    uppern = parameters_instance.uppern
    Exdata = parameters_instance.Exdata
    timelist = parameters_instance.timelist
    Exdata2 = Exdata(timelist[1:])

    files = [f for f in os.listdir() if f.endswith('.dat') and 'fs' not in f]
    error = []
    Avalue = []
    Bvalue = []
    for f in files:
        datdata = open(f, "r", encoding='utf-8').readlines()
        xRF, yRF = np.array(DataRF1(datdata, uppern), dtype=float)
        RFdatas = np.sqrt(np.array(xRF) ** 2 + np.array(yRF) ** 2)
        wlist0 = [timelist[i] - timelist[i - 1] for i in range(2, len(timelist))]
        wlist = np.append(wlist0, wlist0[-1]) / (timelist[-1] - timelist[1])
        RFdatas = np.reshape(RFdatas, (-1, 1))  #### Reshaping and aligning the data
        Exdata2 = np.resize(Exdata2, RFdatas.shape)  #### Reshaping and aligning the data
        error0 = (np.array(Exdata2) - np.array(RFdatas)) ** 2
        error_value = np.sqrt(np.sum([e0 * w for e0, w in zip(error0, wlist)]))

        Bvalue_value = os.path.splitext(f)[0].split('_')[-1]
        Avalue_value = os.path.splitext(f)[0].split('_')[-3]
        print("A:", Avalue_value, ", B:", Bvalue_value, "Error:", error_value)
        error.append(float(error_value))
        Avalue.append(float(Avalue_value))
        Bvalue.append(float(Bvalue_value))
    return error, Avalue, Bvalue

def Best_result(error, Nvalue, Avalue):
    combined = list(zip(error, Nvalue, Avalue))
    min_error = min(combined, key=lambda x: x[0])
    Best_error, Best_Nvalue, Best_Avalue = min_error
    return  Best_error, Best_Nvalue, Best_Avalue

def move_files(subfolder_name, Best_Nvalue):
    subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), subfolder_name)
    Best_Nvalue = float(Best_Nvalue)
    TypeName = "_" + subfolder_name.lower()
    TypeName2 = "_" + subfolder_name.upper()
    ResultName = "Results_" + subfolder_name
    os.makedirs(subfolder, exist_ok=True)
    files_to_move = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if TypeName in f or TypeName2 in f]
    files_to_copy = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if ResultName in f]
    for file in files_to_move:
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        destination_path = os.path.join(subfolder, file)
        shutil.move(source_path, destination_path)
    for file in files_to_copy:
        source_path = os.path.join(subfolder, file)
        destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        shutil.copy2(source_path, destination_path)
    if subfolder_name == 'fs':
        Best_fs_dat = "_fs_" + str(Best_Nvalue) + ".dat"
        fs_to_copy = [f for f in os.listdir(subfolder) if str(Best_fs_dat) in f]
        for file in fs_to_copy:
            source_path = os.path.join(subfolder, file)
            destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            shutil.copy2(source_path, destination_path)
            datdata = open(file, "r", encoding='utf-8').readlines()
            triaxiality = DAT_values(datdata, 19, 2)
            eqplas = DAT_values(datdata, 13, 5)
            ATRI = average_values(triaxiality, eqplas)
            print("                                           ATRI", ATRI)
    elif subfolder_name == 'b':
        Best_csv = "_B_" + str(Best_Nvalue) + "_Load.csv"
        Best_dat = "_B_" + str(Best_Nvalue) + ".dat"
        fs_to_copy = [f for f in os.listdir(subfolder) if str(Best_dat) in f or str(Best_csv) in f]
        for file in fs_to_copy:
            source_path = os.path.join(subfolder, file)
            destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            shutil.copy2(source_path, destination_path)
    return

def move_files_to_subfolder(subfolder_name, error, Nvalue, Avalue):
    subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), subfolder_name)
    TypeName = "_" + subfolder_name
    TypeName2 = "_" + subfolder_name.upper()
    ResultName = "Results_" + subfolder_name
    os.makedirs(subfolder, exist_ok=True)
    files_to_move = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if TypeName in f or TypeName2 in f]
    files_to_copy = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if ResultName in f]
    for file in files_to_move:
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        destination_path = os.path.join(subfolder, file)
        shutil.move(source_path, destination_path)
    for file in files_to_copy:
        source_path = os.path.join(subfolder, file)
        destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        shutil.copy2(source_path, destination_path)
    Best_error, Best_Nvalue, Best_Avalue = Best_result(error, Nvalue, Avalue)
    if subfolder_name == 'n':
        print("BestResult: ", subfolder_name, ":", Best_Nvalue, end=" ")
    elif subfolder_name == 'fs':
        print("BestResult: ", subfolder_name, ":", Best_Nvalue, end=" ")
        Best_fs_dat = "_fs_" + str(Best_Nvalue) + ".dat"
        Best_fs_inp = "_fs_" + str(Best_Nvalue)
        fs_to_copy = [f for f in os.listdir(subfolder) if str(Best_fs_dat) in f]
        fs_to_copy2 = [f for f in os.listdir(subfolder) if str(Best_fs_inp) in f]

        for file in fs_to_copy2:
            source_path = os.path.join(subfolder, file)
            destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            shutil.copy2(source_path, destination_path)
        for file in fs_to_copy:
            destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            datdata = open(destination_path, "r", encoding='utf-8').readlines()
            triaxiality = DAT_values(datdata, 19, 2)
            eqplas = DAT_values(datdata, 13, 5)
            ATRI = average_values(triaxiality, eqplas)
            Best_Avalue = ATRI
            print("    ATRI", ATRI)
    elif subfolder_name == 'b':
        print("BestResult:  B :", Best_Nvalue, "A :", Best_Avalue, end=" ")
        Best_inp = "_B_" + str(Best_Nvalue)
        B_to_copy = [f for f in os.listdir(subfolder) if str(Best_inp) in f]
        for file in B_to_copy:
            source_path = os.path.join(subfolder, file)
            destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            shutil.copy2(source_path, destination_path)
    return float(Best_Nvalue), float(Best_Avalue), float(Best_error)

#### Common part for GoldenMethod
def exjob(x, parameters_instance):
    x = float("{:.6g}".format(x))
    def find_dat_move_1(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance):

        if os.path.exists(Case_dat_name):
            time.sleep(1)
        elif os.path.exists(os.path.join(subfolder, Case_dat_name)) :
            files_to_move = [f for f in os.listdir(subfolder) if f"{Case_name_x}{x}" in f]
            for file in files_to_move:
                source_path = os.path.join(subfolder, file)
                destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
                shutil.move(source_path, destination_path)
            time.sleep(1)
        elif os.path.exists(os.path.join(subfolder2, Case_dat_name)):
            files_to_move = [f for f in os.listdir(subfolder2) if f"{Case_name_x}{x}" in f]
            for file in files_to_move:
                source_path = os.path.join(subfolder2, file)
                destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
                shutil.move(source_path, destination_path)
            time.sleep(1)
        else:
            if identification_type == 'n':
                Tensileinpmaker(x, parameters_instance)
                TensileAbaqusdo(x, parameters_instance)
                time.sleep(30)
            elif identification_type == 'fs':
                FSTensileinpmaker(x, parameters_instance)
                FSTensileAbaqusdo(x, parameters_instance)
                time.sleep(30)
            elif identification_type == 'b':
                Arcaninpmaker_B(x, parameters_instance)
                ArcanAbaqusdo_B(x, parameters_instance)
                time.sleep(30)

        if identification_type == 'n':
            Tensileendlog(x, parameters_instance)
            Nerror = Tensileerrormin(x, parameters_instance)
        elif identification_type == 'fs':
            Tensileendlog(x, parameters_instance)
            Nerror = FSTensileerrormin(x, parameters_instance)
        elif identification_type == 'b':
            Arcanendlog(x, parameters_instance)
            Nerror = Arcanerrormin_B(x, parameters_instance)
        return Nerror

    identification_type = parameters_instance.identification_type
    subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), identification_type)
    subfolder2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), identification_type, 'useless')
    if identification_type == 'n':
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{x}.dat"
        Nerror = find_dat_move_1(Case_dat_name, subfolder, subfolder2, Case_name, x, identification_type, parameters_instance)
    elif identification_type == 'fs':
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{x}.dat"
        Nerror = find_dat_move_1(Case_dat_name, subfolder, subfolder2, Case_name, x, identification_type, parameters_instance)
    elif identification_type == 'b':
        fs = parameters_instance.fs
        ATRI = parameters_instance.ATRI
        A = ABCfuction(x,fs,ATRI)
        A = float("{:.6g}".format(A))
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{A}_B_{x}.dat"
        Case_name_x = f"{Case_name}{A}_B_"
        Nerror = find_dat_move_1(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance)
    return Nerror

def exjob2(x, x2, parameters_instance):
    x = float("{:.6g}".format(x))
    x2 = float("{:.6g}".format(x2))
    # print("Predicted condition: ", x, x2)
    def find_dat_move(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance):
        if os.path.exists(Case_dat_name):
            time.sleep(1)
        elif os.path.exists(os.path.join(subfolder, Case_dat_name)) :
            files_to_move = [f for f in os.listdir(subfolder) if f"{Case_name_x}{x}" in f]
            for file in files_to_move:
                source_path = os.path.join(subfolder, file)
                destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
                shutil.move(source_path, destination_path)
            time.sleep(1)
        elif os.path.exists(os.path.join(subfolder2, Case_dat_name)):
            files_to_move = [f for f in os.listdir(subfolder2) if f"{Case_name_x}{x}" in f]
            for file in files_to_move:
                source_path = os.path.join(subfolder2, file)
                destination_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
                shutil.move(source_path, destination_path)
            time.sleep(1)
        else:
            if identification_type == 'n':
                Tensileinpmaker(x, parameters_instance)
                TensileAbaqusdo(x, parameters_instance)
                time.sleep(10)
            elif identification_type == 'fs':
                FSTensileinpmaker(x, parameters_instance)
                FSTensileAbaqusdo(x, parameters_instance)
                time.sleep(10)
            elif identification_type == 'b':
                Arcaninpmaker_B(x, parameters_instance)
                ArcanAbaqusdo_B(x, parameters_instance)
                time.sleep(10)
        return

    identification_type = parameters_instance.identification_type
    subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), identification_type)
    subfolder2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), identification_type, 'useless')
    if identification_type == 'n':
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{x}.dat"
        Case_dat_name2 =f"{Case_name}{x2}.dat"
        Case_name_x = f"{Case_name}"
        find_dat_move(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance)
        find_dat_move(Case_dat_name2, subfolder, subfolder2, Case_name_x, x2, identification_type, parameters_instance)
    elif identification_type == 'fs':
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{x}.dat"
        Case_dat_name2 = f"{Case_name}{x2}.dat"
        Case_name_x = f"{Case_name}"
        find_dat_move(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance)
        find_dat_move(Case_dat_name2, subfolder, subfolder2, Case_name_x, x2, identification_type, parameters_instance)
    elif identification_type == 'b':
        fs = parameters_instance.fs
        ATRI = parameters_instance.ATRI
        A = ABCfuction(x,fs,ATRI)
        A = float("{:.6g}".format(A))
        A2 = ABCfuction(x2,fs,ATRI)
        A2 = float("{:.6g}".format(A2))
        Case_name = parameters_instance.Case_name
        Case_dat_name = f"{Case_name}{A}_B_{x}.dat"
        Case_dat_name2 = f"{Case_name}{A2}_B_{x2}.dat"
        Case_name_x = f"{Case_name}{A}_B_"
        Case_name_x2 = f"{Case_name}{A2}_B_"
        find_dat_move(Case_dat_name, subfolder, subfolder2, Case_name_x, x, identification_type, parameters_instance)
        find_dat_move(Case_dat_name2, subfolder, subfolder2, Case_name_x2, x2, identification_type, parameters_instance)
    return

def terminate(x, parameters_instance):
    x = float("{:.6g}".format(x))
    identification_type = parameters_instance.identification_type
    with open("terminate.bat", "r") as terminate_file:
        terminate023 = terminate_file.readlines()

    if identification_type == 'n':
        Case_name = parameters_instance.Case_name
        TypeName = f"{Case_name}{x}"
        Case_bat_name = f"mterminate_{TypeName}.bat"
        Case_log_name = f"{TypeName}.log"
        terminate023[5] = f"ab2023 terminate job={TypeName}\n"
    elif identification_type == 'fs':
        Case_name = parameters_instance.Case_name
        TypeName = f"{Case_name}{x}"
        Case_bat_name = f"mterminate_{TypeName}.bat"
        Case_log_name = f"{TypeName}.log"
        terminate023[5] = f"ab2023 terminate job={TypeName}\n"
    elif identification_type == 'b':
        fs = parameters_instance.fs
        ATRI = parameters_instance.ATRI
        A = ABCfuction(x,fs,ATRI)
        A = float("{:.6g}".format(A))
        Case_name = parameters_instance.Case_name
        TypeName = f"{Case_name}{A}_B_{x}"
        Case_bat_name = f"mterminate_{TypeName}.bat"
        Case_log_name = f"{TypeName}.log"
        terminate023[5] = f"ab2023 terminate job={TypeName}\n"

    if os.path.exists(Case_log_name):
        with open(Case_log_name, "r") as logfile:
            lines = logfile.readlines()
        fin = [i for i, line in enumerate(lines) if "COMPLETED" in line]
        fin2 = [i for i, line in enumerate(lines) if "Wrap-up" in line]
        if not fin and not fin2:
            with open(Case_bat_name, "w") as terminate_output:
                terminate_output.writelines(terminate023)
            time.sleep(5)
            try:
                subprocess.Popen(["cmd.exe", "/c", f"start {Case_bat_name}"], shell=True)
                time.sleep(120)
            except Exception as e:
                print("Error:", e)
        subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), identification_type, 'useless')
        os.makedirs(subfolder, exist_ok=True)
        files_to_move = [f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if TypeName in f]
        for file in files_to_move:
            source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            destination_path = os.path.join(subfolder, file)
            shutil.move(source_path, destination_path)
    return

def GoldenMethod1Parallel(parameters_instance, jobnumber):
    u = parameters_instance.u
    l = parameters_instance.l
    residualerror = parameters_instance.residualerror
    print("Iterative calculation started", np.datetime64('now'))
    s = 0.381966  # (Sqrt[5]-1)/(Sqrt[5]+1)
    b0 = u - l
    t0 = s * b0
    x0 = l
    x1 = l + t0
    x2 = u - t0
    x3 = u
    savelist = []
    if jobnumber==3:
        exjob2(x1, x2, parameters_instance)
    p1 = exjob(x1, parameters_instance)
    p2 = exjob(x2, parameters_instance)

    n = 0
    kx0 = x0
    kx1 = x1
    kx2 = x2
    kx3 = x3

    if p1 < p2:
        x0 = kx0
        x1 = kx2 - t0
        x2 = kx1
        x3 = kx2
        j = 1
    else:
        x0 = kx1
        x1 = kx2
        x2 = kx1 + t0
        x3 = kx3
        j = 2

    n += 1
    b = b0 - t0
    t = s * b
    savelist.append([x0, x1, x2, x3, b, t, j])

    while b > residualerror:
        kx0 = x0
        kx1 = x1
        kx2 = x2
        kx3 = x3

        if j == 1:
            if jobnumber == 3:
                exjob2(kx1 + t, kx2 - t, parameters_instance)
                kp1 = exjob(kx1, parameters_instance)
                print(f"Predicted condition: {kx1 + t:.6f}, {kx2 - t:.6f}")
            else:
                kp1 = exjob(kx1, parameters_instance)
            kp2 = p1

        else:
            if jobnumber == 3:
                exjob2(kx1 + t, kx2 - t, parameters_instance)
                kp2 = exjob(kx2, parameters_instance)
                print("Predicted condition: ", kx1 + t, kx2 - t)
            else:
                kp2 = exjob(kx2, parameters_instance)
            kp1 = p2


        if kp1 < kp2:
            if jobnumber == 3:
                terminate(kx1 + t, parameters_instance)
            x0 = kx0
            x1 = kx2 - t
            x2 = kx1
            x3 = kx2
            p1 = kp1
            j = 1
        else:
            if jobnumber == 3:
                terminate(kx2 - t, parameters_instance)
            x0 = kx1
            x1 = kx2
            x2 = kx1 + t
            x3 = kx3
            p2 = kp2
            j = 2

        b = b - t
        t = s * b
        n += 1
        savelist.append([x0, x1, x2, x3, b, t, j])
    if jobnumber == 3:
        terminate(kx2 - t, parameters_instance)
        terminate(kx1 + t, parameters_instance)

    print("Finished", np.datetime64('now'))

#### Common part initialize
def Identification(identification_parameter, material):
    identification_parameter = identification_parameter.lower()
    if identification_parameter == 'n':
        identification_type = 'n'
    elif identification_parameter == 'fs':
        identification_type = 'fs'
    elif identification_parameter == 'b':
        identification_type = 'b'
    else:
        print("ERROR in identification parameter")
    if material == 'AL':
        print("MATERIAL:",material)
        RFarea = 15
        d = 25
    elif material == 'ST':
        print("MATERIAL:",material)
        RFarea = 25 * 0.71
        d = 18
    else:
        print("ERROR in material type")
    return identification_type,RFarea,d

def initialize_parameters_all():
    dstep1 = 4.8
    thickness = 0.6
    nz = 3
    return thickness, nz, dstep1

def initialize_parameters(name):
    if name=="n":
        ftstep2 = 0.01
        dtstep2 = 0.01
        dstep2 = 7.5
    elif name=="fs":
        ftstep2 = 0.0067
        dtstep2 = 0.0067
        dstep2 = 9
    elif name=="b":
        ftstep2 = 0.01
        dtstep2 = 0.01
        dstep2 = 7.5
    else:
        ftstep2 = 0.01
        dtstep2 = 0.01
        dstep2 = 7.5
    return ftstep2, dtstep2, dstep2

def DAT_values(data, SDVN, n):
    SDV_keyword = "SDV"+ str(SDVN)
    a = [i for i, line in enumerate(data) if SDV_keyword in line]
    b = list(range(1, len(a) + 1))
    DAT_value_all = [data[a[i] + 3].split()[n] for i in range(len(b))]
    DAT_value = []
    for value in DAT_value_all :
        try:
            num = float(value)
            DAT_value.append(num)
        except ValueError:
            pass
    return DAT_value

def average_values(triaxiality, eqplas):
    results = np.transpose([eqplas, triaxiality])
    results = np.unique(results, axis=0)[:-1][1:]
    dydx = np.gradient(results[:, 1]) / np.gradient(results[:, 0])
    interp_func = CubicHermiteSpline(results[:, 0], results[:, 1], dydx)
    integral_result, _ = quad(interp_func, np.min(results[:, 0]), np.max(results[:, 0]), limit=1000)
    # Method1
    # from scipy.interpolate import PchipInterpolator
    # interp_func = PchipInterpolator(results[:, 0], results[:, 1])
    # interp_func = interp1d(results[:, 0], results[:, 1])
    # Method2
    # def integrate_segment(interp_func, x_min, x_max, num_segments):
    #     segment_width = (x_max - x_min) / num_segments
    #     total_integral = 0.0
    #     for i in range(num_segments):
    #         segment_start = x_min + i * segment_width
    #         segment_end = segment_start + segment_width
    #         integral, _ = quad(interp_func, segment_start, segment_end)
    #         total_integral += integral
    #     return total_integral
    # num_segments = 10
    # integral_result = integrate_segment(interp_func, np.min(results[:, 0]), np.max(results[:, 0]), num_segments)
    max_value = np.max(results[:, 0])
    result = integral_result / max_value
    return result

def SDVlist_parameter():
    SDVnamelist0 = ["EELAS(1)", "EELAS(2)", "EELAS(3)", "EELAS(4)", "EELAS(5)", "EELAS(6)",
                    "EPLAS(1)", "EPLAS(2)", "EPLAS(3)", "EPLAS(4)", "EPLAS(5)", "EPLAS(6)",
                    "EQPLAS", "DAMAGE", "SYIELD", "FFLAG", "DEQPL", "SYIELDMAX", "TRIXIALITY",
                    "LODE ANGLE", "ID", "CRITICAL PLASTIC STRAIN", "SEQPLAS", "IFFLAG", "ID3",
                    "JSLOPE"]
    SDVlist = [f"{i},{name}" for i, name in enumerate(SDVnamelist0, 1)]
    iSDVf = SDVnamelist0.index("FFLAG") + 1
    nSDV = len(SDVnamelist0)
    odb=["MISESONLY", "S", "SDV13", "SDV14", "SDV16", "SDV19", "SDV20"]
    return nSDV, SDVlist, iSDVf, odb

def ABCfuction(x,fs,ATRI):
    A = fs / np.exp(-x * ATRI)
    return A

# # -------------------------Starting Calculation--------------------------------------------------
def START_NFsB(identification_parameter, material, inpname, inpname2,
               dstep1, ftstep2, dtstep2, dstep2,u, l, UMATname,
               nn, fs, nz, thickness, cpus, jobnumber):
    ftstep1 = 0.001
    dtstep1 = 0.1
    frestep1 = 10
    frestep2 = 10
    residualerror = 0.01
    bline = 300
    averageformer = 250
    averagelatter = 30

    elementtype = "C3D8"
    numatmpinc = 10
    incnum = 10000  # maximum number of increment
    finc = 1  # first increment
    mininc = 1.0e-7  # minimum increment
    maxinc = 1  # maximum increment
    identification_type, RFarea, d = Identification(identification_parameter, material)
    if identification_type == 'n':
        print("-----------------Start N identification-----------------")
        Hinpmaker(material, inpname, ftstep1, dtstep1, dstep1, frestep1, ftstep2, dtstep2, dstep2, frestep2)
        parameters_instance = ParametersProcessing(material, inpname, UMATname, u, l, RFarea, d, identification_type,
                                                   bline, averageformer, averagelatter, cpus, residualerror)
        GoldenMethod1Parallel(parameters_instance, jobnumber)
        exp, expi = From_experiment(identification_type, material)
        error, Nvalue = alldatafind(exp, expi, parameters_instance, identification_type)
        Best_Nvalue, Best_Avalue, Best_error = move_files_to_subfolder(identification_type, error, Nvalue, [0] * len(Nvalue))
        # Avalue=0.0
        data = [[f'{identification_parameter}', 'Error']] + list(zip(Nvalue, error))
        best_data = [["", ""], [f'Best_value', f'Best_error'], [f'{Best_Nvalue:.6f}', f'{Best_error:.6f}'], ["OVER"]]
        identification_type = 'fs'
    elif identification_type == 'fs':
        print("-----------------Start Fs identification-----------------")
        parameters_instance = FSParametersProcessing(material, inpname, UMATname, u, l, RFarea, d, identification_type, nn,
                                           bline, averageformer, averagelatter, cpus, residualerror)
        FSinpmaker(material, inpname, ftstep1, dtstep1, dstep1, frestep1, ftstep2, dtstep2, dstep2, frestep2)
        GoldenMethod1Parallel(parameters_instance, jobnumber)
        exp, expi = From_experiment(identification_type, material)
        error, Nvalue = alldatafind(exp, expi, parameters_instance, identification_type)
        Best_Nvalue, Best_Avalue, Best_error = move_files_to_subfolder(identification_type, error, Nvalue, [0] * len(Nvalue))
        # Avalue = 0.0
        data = [[f'{identification_parameter}', 'Error']] + list(zip(Nvalue, error))
        best_data = [["", "", ""], [f'Best_value', f'Best_error', f'ATRI'],
                     [f'{Best_Nvalue:.6f}', f'{Best_error:.6f}', f'{Best_Avalue:.6f}'], ["OVER"]]
        identification_type = 'b'
    elif identification_type == 'b':
        print("-----------------Start B identification-----------------")
        name = inpname2   #material + "_arcan"
        thresold = 0.0125
        iout = 5
        ArcanMesh(name, material, thickness, nz, elementtype, bline, averageformer, averagelatter, nn)
        parameters_instance =ArcanStep(name, material, inpname, fs, iout, thresold, incnum, finc, mininc, maxinc, numatmpinc,
                      elementtype, u, l, cpus, UMATname, residualerror, identification_type)
        GoldenMethod1Parallel(parameters_instance, jobnumber)
        error, Avalue, Nvalue = allArcandatafind(parameters_instance)
        Best_Nvalue, Best_Avalue, Best_error = move_files_to_subfolder(identification_type, error, Nvalue, Avalue)
        data = [['A', f'{identification_parameter}', 'Error']] + list(zip(Avalue, Nvalue, error))
        best_data = [["", "", ""], [f'Best_valueA', f'Best_valueB', f'Best_error'],
                     [f'{Best_Avalue:.6f}', f'{Best_Nvalue:.6f}', f'{Best_error:.6f}'], ["OVER"]]
        identification_type = 'OVER'
    with open(f'Results_{identification_parameter.lower()}.csv', "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
        csvwriter.writerows(best_data)
    return identification_type, Best_Nvalue

def ALL_START_NFsB(material, inpname, inpname2, UMATname, dstep1,
                    ftstep2n, dtstep2n, dstep2n, un, ln,
                    ftstep2fs, dtstep2fs, dstep2fs, ufs, lfs,
                    ftstep2b, dtstep2b, dstep2b, ub, lb,
                    nz, thickness, cpus, jobnumber):
    def ALL_NFsB():
        def index(para):
            if os.path.exists(f"Results_{para}.csv") or (
                    os.path.exists("N") and os.path.exists(os.path.join(f"{para}", f"Results_{para}.csv"))):
                if os.path.exists(f"Results_{para}.csv"):
                    with open(f"Results_{para}.csv", "r") as file:
                        content = file.readlines()
                elif os.path.exists(os.path.join(f"{para}", f"Results_{para}.csv")):
                    with open(os.path.join(f"{para}", f"Results_{para}.csv"), "r") as file:
                        content = file.readlines()
                content = [line.strip().split(',') for line in content]
                found_over = False
                for i in range(1, len(content)):
                    if 'OVER' in content[i][0]:  # If "Over" appears in the second column.
                        found_over = True
                        Best_Value = content[i - 1]
                        break
                if found_over:
                    best = 1
                    print(f"{para:2s} Identification Already Existed         ", end=" ")
                    if para.lower() == "b":
                        print(f"Best:  A  = {Best_Value[0]},B = {Best_Value[1]} , Error = {Best_Value[2]}")
                        Bvalue = Best_Value[1]
                        Avalue = Best_Value[0]
                        return best, Avalue, Bvalue
                    elif para.lower() == "fs":
                        print(f"Best:  {para.upper():2s} = {Best_Value[0]}              , Error = {Best_Value[1]}")
                        value = Best_Value[0]
                        value2 = 0.0
                        return best, value, value2
                    else:
                        print(f"Best:  {para.upper():2s} = {Best_Value[0]}              , Error = {Best_Value[1]}")
                        value = Best_Value[0]
                        value2 = 0.0
                        return best, value, value2
                else:
                    print(f"{para} identification existed but did not finish yet")
                    best = 0
                    value1 = 0.0
                    value2 = 0.0
                    return best, value1, value2
            else:
                print(f"{para} identification did not exist yet")
                best = 0
                value1 = 0.0
                value2 = 0.0
                return best, value1, value2
        identification_parameter = 'n'
        index_N, nn, fs = index(identification_parameter)
        if index_N == 1:
            move_files(identification_parameter, nn)
            identification_parameter = 'fs'
            index_fs, fs, ATRI = index(identification_parameter)
            if index_fs == 1:
                move_files(identification_parameter, fs)
                identification_parameter = 'b'
                index_B, Avalue, Bvalue = index(identification_parameter)
                if index_B == 1:
                    move_files(identification_parameter, Bvalue)
                    identification_parameter = 'OVER'
        return identification_parameter, float(nn), float(fs)
    identification_parameter, nn, fs = ALL_NFsB()
    if identification_parameter == "n":
        identification_parameter, nn = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                       dstep1, ftstep2n, dtstep2n, dstep2n, un, ln, UMATname,
                                                       nn, fs, nz, thickness, cpus, jobnumber)
        identification_parameter, fs = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                  dstep1, ftstep2fs, dtstep2fs, dstep2fs, ufs, lfs, UMATname,
                                                  nn, fs, nz, thickness, cpus, jobnumber)
        identification_parameter, b = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                 dstep1, ftstep2b, dtstep2b, dstep2b, ub, lb, UMATname,
                                                 nn, fs, nz, thickness, cpus, jobnumber)
        print("Finished!!!!")
    elif identification_parameter == "fs":
        identification_parameter, fs = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                  dstep1, ftstep2fs, dtstep2fs, dstep2fs, ufs, lfs, UMATname,
                                                  nn, fs, nz, thickness, cpus, jobnumber)
        identification_parameter, b = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                 dstep1, ftstep2b, dtstep2b, dstep2b, ub, lb, UMATname,
                                                 nn, fs, nz, thickness, cpus, jobnumber)
        print("Finished!!!!")
    elif identification_parameter == "b":
        identification_parameter, b = START_NFsB(identification_parameter, material, inpname, inpname2,
                                                 dstep1, ftstep2b, dtstep2b, dstep2b, ub, lb, UMATname,
                                                 nn, fs, nz, thickness, cpus, jobnumber)
        print("Finished!!!!")
    elif identification_parameter == "OVER":
        print("Finished!!!!")
    else:
        print("Wrong!!!")

    subfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ALLResults')
    subfolder2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ALLInputs')
    source_folder = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(subfolder, exist_ok=True)
    os.makedirs(subfolder2, exist_ok=True)

    def pkl_csv_exist(filename, source_folder):
        subfolder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n', filename)
        subfolder_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fs', filename)
        subfolder_path3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b', filename)
        subfolder_path4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ALLResults', filename)
        if os.path.exists(filename):
            print("")
        elif os.path.exists(os.path.join(subfolder_path)):
            shutil.copy(subfolder_path, source_folder)
        elif os.path.exists(os.path.join(subfolder_path2)):
            shutil.copy(subfolder_path2, source_folder)
        elif os.path.exists(os.path.join(subfolder_path3)):
            shutil.copy(subfolder_path3, source_folder)
        elif os.path.exists(os.path.join(subfolder_path4)):
            shutil.copy(subfolder_path4, source_folder)
        return

    pkl_csv_exist('Data_N.pkl', source_folder)
    pkl_csv_exist('Data_fs.pkl', source_folder)
    pkl_csv_exist('Data_B.pkl', source_folder)
    pkl_csv_exist('Results_n.csv', source_folder)
    pkl_csv_exist('Results_fs.csv', source_folder)
    pkl_csv_exist('Results_b.csv', source_folder)

    with open("Results_n.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data_n = []
        error_n = []
        for row in reader:
            try:
                data_n.append(float(row[0]))
                error_n.append(float(row[1]))
            except:
                next(reader)
                break
        Best_n = next(reader)
        Best_result_n = float(Best_n[0])
        Best_error_n = float(Best_n[1])
    with open("Results_fs.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data_fs = []
        error_fs = []
        for row in reader:
            try:
                data_fs.append(float(row[0]))
                error_fs.append(float(row[1]))
            except:
                next(reader)
                break
        Best_fs = next(reader)
        Best_result_fs = float(Best_fs[0])
        Best_error_fs = float(Best_fs[1])
        Best_result_ATRI = float(Best_fs[2])
    with open("Results_b.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data_a = []
        data_b = []
        error_b = []
        for row in reader:
            try:
                data_a.append(float(row[0]))
                data_b.append(float(row[1]))
                error_b.append(float(row[2]))
            except:
                next(reader)
                break
        Best_b = next(reader)
        Best_result_a = float(Best_b[0])
        Best_result_b = float(Best_b[1])
        Best_error_b = float(Best_b[2])
    with open("Results_ALL.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Best_n", "Best_fs", "Best_atri", "Best_a", "Best_b"])
        writer.writerow([Best_result_n, Best_result_fs, Best_result_ATRI, Best_result_a, Best_result_b])
        writer.writerow([])
        writer.writerow([])
        output_data = [data_n, error_n, data_fs, error_fs, data_a, data_b, error_b]
        writer.writerow(["n", "error_n", "fs", "error_fs", "a", "b", "error_b"])
        for row in zip_longest(*output_data):
            writer.writerow(row)

    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        if "_fs" in filename or "_b" in filename or "_B" in filename or "_n" in filename or "_N" in filename or "_mesh" in filename or "_A.inp" in filename:
            shutil.move(source_file, subfolder)
        elif ".py" in filename or "ST" in filename or ".csv" in filename or ".bat" in filename or "02" in filename or ".f" in filename:
            if "_ALL.csv" not in filename:
                shutil.move(source_file, subfolder2)
    print("Finished!!!!")
    return



