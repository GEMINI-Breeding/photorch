import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from photorch.src.fvcb.evaluate import evaluateFvCB
from photorch.src.stomatal.evaluate import evaluateBMF
from photorch.src import stomatal
from photorch.src import fvcb
import torch

def computeR2(obs, pred):
    return 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def computeRMSE(obs,pred):
    return (np.sum((obs-pred)**2)/len(obs)) ** 0.5

def compileACiFiles(path):
    # Walk through directory with ACi curves and compile file paths to the LI-6800 plain text files (those without extensions)
    curvefiles = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if not file.endswith(".xlsx"):
                if file[-1].isdigit():
                    full_path = os.path.join(root,file)
                    curvefiles.append(full_path)

    # Add .txt to files
    for file_path in curvefiles:
        print(file_path)

    for filename in curvefiles:
        if '.' not in filename: 
            new_filename = filename + '.txt'
            print(new_filename)
            os.rename(filename,new_filename)

    # Verify files to be fit into ACi fitter
    curvefiles = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root,file)
                curvefiles.append(full_path)

    for i, file in enumerate(curvefiles):
        if(i==0):
            print("Files compiled:")
        print(file)

    # Concatenate all A-Ci Curve Data into a Dataframe
    rows_to_skip = 66            # Number of rows to skip at the top of LI-6800 file (header data)
    all_data = pd.DataFrame()

    for i, infile in enumerate(curvefiles):

        df = pd.read_csv(infile, skiprows=rows_to_skip, delimiter="\t")
        df = df.drop(0) # Drop Units row
        df = df.loc[:, ['A', 'Ci', 'Qin', 'Tleaf']]
        df[['A', 'Ci', 'Qin', 'Tleaf']] = df[['A', 'Ci', 'Qin', 'Tleaf']].apply(pd.to_numeric, errors='coerce').fillna(0) # Clean, fill NaNs
        df['CurveID'] = i + 1
        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data['FittingGroup'] = 1
    all_data.to_csv(path+"/curves.csv", index=False)
    print(f'Saved compiled curves to {path+"/curves.csv"}.')
    return path+"/curves.csv"

def normalizeACiCurveGroupsWithSurvey(curveGroupPaths,surveyDataPath):
    return
    # Apply a Renormalization to A-Ci Curve Groups According to an Average of a Sampled Amax of the Fitting Group
    # Curve groups are those curves which occured on different leaves and need to be normalized
    # import os
    # import pandas as pd
    # import numpy as np
    # survey_path = "data/raw/aci/survey/iceberg"
    # for root,dirs,files in os.walk(survey_path):
    #     for file in files:
    #         if not file.endswith(".xlsx"):
    #             if ("survey" in file.lower()) & (not file.endswith(".txt")):
    #                 print(file)
    #                 full_path = os.path.join(root,file)
    #                 os.rename(full_path,full_path+".txt")
    # for root,dirs,files in os.walk(survey_path):
    #     for file in files:
    #         if not file.endswith(".xlsx"):
    #             if ("survey" in file.lower()) & (file.endswith(".txt")):
    #                 full_path = os.path.join(root,file)
    #                 print(full_path)

    # df = pd.read_csv(full_path, skiprows=66, delimiter="\t")
    # df = df.drop(0) # Drop Units row
    # A_avg = df["A"].apply(pd.to_numeric, errors='coerce').fillna(0).mean()
    # A_max=0
    # for grp in np.unique(all_data.CurveID.loc[all_data.CurveID<7]):
    #     group_Q_avg = all_data.Qin.loc[all_data.CurveID==grp].mean()
    #     group_T_avg = all_data.Tleaf.loc[all_data.CurveID==grp].mean()
    #     group_A_max = all_data.A.loc[all_data.CurveID==grp].max()
    #     A_max = max(A_max,group_A_max)
    #     print(A_max)
    #     print(grp," ",all_data.A.loc[all_data.CurveID==grp].max()," ",all_data.Qin.loc[all_data.CurveID==grp].max()," ",all_data.Tleaf.loc[all_data.CurveID==grp].max())

    # norm = A_avg/A_max
    # all_data.loc[all_data.CurveID<7,"A"] = all_data.loc[all_data.CurveID<7,"A"]*norm

    # A_max=0
    # for grp in np.unique(all_data.CurveID.loc[all_data.CurveID>=7]):
    #     group_A_max = all_data.A.loc[all_data.CurveID==grp].max()
    #     A_max = max(A_max,group_A_max)
    #     print(A_max)
    #     print(grp," ",all_data.A.loc[all_data.CurveID==grp].max()," ",all_data.Qin.loc[all_data.CurveID==grp].max()," ",all_data.Tleaf.loc[all_data.CurveID==grp].max())

    # norm = A_avg/A_max
    # print(norm)
    # print(all_data.A.max())
    # all_data.loc[all_data.CurveID<=7,"A"] = all_data.loc[all_data.CurveID<=7,"A"]+10
    # print(all_data.A.max())

def printFvCBParameters(fvcbm,LightResponseType=1,TemperatureResponseType=2,Fitgm=False,FitGamma=False,FitKc=False,FitKo=False):
    print(f"Vcmax25 = {fvcbm.Vcmax25[0]}")
    print(f"Jmax25 = {fvcbm.Jmax25[0]}")
    print(f"Vcmax_dHa = {fvcbm.TempResponse.dHa_Vcmax[0]}")
    print(f"Jmax_dHa = {fvcbm.TempResponse.dHa_Jmax[0]}")
    print(f"alpha = {fvcbm.LightResponse.alpha[0]}")
    if(LightResponseType==2):
        print(f"theta = {fvcbm.LightResponse.theta[0]}")
    if(Fitgm):
        print(f"gm = {fvcbm.gm[0]}")
    if(FitGamma):
        print(f"Gamma25 = {fvcbm.Gamma25[0]}")
    if(FitKc):
        print(f"Kc = {fvcbm.Kc25[0]}")
    if(FitKo):
        print(f"Ko = {fvcbm.Ko25[0]}")

def saveFvCBParametersToFile(species,var,fvcbm,LightResponseType=1,TemperatureResponseType=2,Fitgm=False,FitGamma=False,FitKc=False,FitKo=False):
    savepath = "results/parameters/"+species+var+"_FvCB_Parameters.csv"
    if(LightResponseType==2 & TemperatureResponseType==2):
        vars = ["species","Vcmax25","Jmax25","TPU25","Rd25","alpha","theta","Vcmax_dHa","Vcmax_Topt","Vcmax_dHd","Jmax_dHa","Jmax_Topt","Jmax_dHd","TPU_dHa","TPU_Topt","TPU_dHd","Rd_dHa","Gamma25","Gamma_dHa","Kc25","Kc_dHa","Ko25","Ko_dHa","O"]
        vals = [species,fvcbm.Vcmax25.item(),fvcbm.Jmax25.item(),fvcbm.TPU25.item(),fvcbm.Rd25.item(),fvcbm.LightResponse.alpha.item(),fvcbm.LightResponse.theta.item(),fvcbm.TempResponse.dHa_Vcmax.item(),fvcbm.TempResponse.Topt_Vcmax.item(),fvcbm.TempResponse.dHd_Vcmax.item(),fvcbm.TempResponse.dHa_Jmax.item(),fvcbm.TempResponse.Topt_Jmax.item(),fvcbm.TempResponse.dHd_Jmax.item(),fvcbm.TempResponse.dHa_TPU.item(),fvcbm.TempResponse.Topt_TPU.item(),fvcbm.TempResponse.dHd_TPU.item(),fvcbm.TempResponse.dHa_Rd.item(),fvcbm.Gamma25.item(),fvcbm.TempResponse.dHa_Gamma.item(),fvcbm.Kc25.item(),fvcbm.TempResponse.dHa_Kc.item(),fvcbm.Ko25.item(),fvcbm.TempResponse.dHa_Ko.item(),fvcbm.Oxy.item()]
        outdf = pd.DataFrame([vals],columns=vars)
        outdf.to_csv(savepath,index=False)
    elif(LightResponseType==2 & TemperatureResponseType==1):
        vars = ["species","Vcmax25","Jmax25","TPU25","Rd25","alpha","theta","Vcmax_dHa","Vcmax_Topt","Vcmax_dHd","Jmax_dHa","Jmax_Topt","Jmax_dHd","TPU_dHa","TPU_Topt","TPU_dHd","Rd_dHa","Gamma25","Gamma_dHa","Kc25","Kc_dHa","Ko25","Ko_dHa","O"]
        vals = [species,fvcbm.Vcmax25.item(),fvcbm.Jmax25.item(),fvcbm.TPU25.item(),fvcbm.Rd25.item(),fvcbm.LightResponse.alpha.item(),fvcbm.LightResponse.theta.item(),fvcbm.TempResponse.dHa_Vcmax.item(),99999,1,fvcbm.TempResponse.dHa_Jmax.item(),99999,1,fvcbm.TempResponse.dHa_TPU.item(),99999,1,fvcbm.TempResponse.dHa_Rd.item(),fvcbm.Gamma25.item(),fvcbm.TempResponse.dHa_Gamma.item(),fvcbm.Kc25.item(),fvcbm.TempResponse.dHa_Kc.item(),fvcbm.Ko25.item(),fvcbm.TempResponse.dHa_Ko.item(),fvcbm.Oxy.item()]
        outdf = pd.DataFrame([vals],columns=vars)
        outdf.to_csv(savepath,index=False)
    elif(LightResponseType==1 & TemperatureResponseType==2):
        vars = ["species","Vcmax25","Jmax25","TPU25","Rd25","alpha","theta","Vcmax_dHa","Vcmax_Topt","Vcmax_dHd","Jmax_dHa","Jmax_Topt","Jmax_dHd","TPU_dHa","TPU_Topt","TPU_dHd","Rd_dHa","Gamma25","Gamma_dHa","Kc25","Kc_dHa","Ko25","Ko_dHa","O"]
        vals = [species,fvcbm.Vcmax25.item(),fvcbm.Jmax25.item(),fvcbm.TPU25.item(),fvcbm.Rd25.item(),fvcbm.LightResponse.alpha.item(),0.0,fvcbm.TempResponse.dHa_Vcmax.item(),fvcbm.TempResponse.Topt_Vcmax.item(),fvcbm.TempResponse.dHd_Vcmax.item(),fvcbm.TempResponse.dHa_Jmax.item(),fvcbm.TempResponse.Topt_Jmax.item(),fvcbm.TempResponse.dHd_Jmax.item(),fvcbm.TempResponse.dHa_TPU.item(),fvcbm.TempResponse.Topt_TPU.item(),fvcbm.TempResponse.dHd_TPU.item(),fvcbm.TempResponse.dHa_Rd.item(),fvcbm.Gamma25.item(),fvcbm.TempResponse.dHa_Gamma.item(),fvcbm.Kc25.item(),fvcbm.TempResponse.dHa_Kc.item(),fvcbm.Ko25.item(),fvcbm.TempResponse.dHa_Ko.item(),fvcbm.Oxy.item()]
        outdf = pd.DataFrame([vals],columns=vars)
        outdf.to_csv(savepath,index=False)
    elif(LightResponseType==1 & TemperatureResponseType==1):
        vars = ["species","Vcmax25","Jmax25","TPU25","Rd25","alpha","theta","Vcmax_dHa","Vcmax_Topt","Vcmax_dHd","Jmax_dHa","Jmax_Topt","Jmax_dHd","TPU_dHa","TPU_Topt","TPU_dHd","Rd_dHa","Gamma25","Gamma_dHa","Kc25","Kc_dHa","Ko25","Ko_dHa","O"]
        vals = [species,fvcbm.Vcmax25.item(),fvcbm.Jmax25.item(),fvcbm.TPU25.item(),fvcbm.Rd25.item(),fvcbm.LightResponse.alpha.item(),0.0,fvcbm.TempResponse.dHa_Vcmax.item(),99999,1,fvcbm.TempResponse.dHa_Jmax.item(),99999,1,fvcbm.TempResponse.dHa_TPU.item(),99999,1,fvcbm.TempResponse.dHa_Rd.item(),fvcbm.Gamma25.item(),fvcbm.TempResponse.dHa_Gamma.item(),fvcbm.Kc25.item(),fvcbm.TempResponse.dHa_Kc.item(),fvcbm.Ko25.item(),fvcbm.TempResponse.dHa_Ko.item(),fvcbm.Oxy.item()]
        outdf = pd.DataFrame([vals],columns=vars)
        outdf.to_csv(savepath,index=False)
    else:
        print(f"LightResponseType={LightResponseType}, TemperatureResponseType={TemperatureResponseType}. They should be either 1 or 2.")
        return
    print(f"Parameters saved to: {savepath}")
    return savepath
    
def plotFvCBModelFit(species,variety,parameterPath,compiledDataPath):
    P = pd.read_csv(parameterPath)
    
    plt.figure(figsize=(10, 4))

    # A vs Ci @ Q2000, T = 35 °C
    Ci = np.linspace(0, 2000, 60)
    T = (35 + 273.15) * np.ones_like(Ci)
    Q = 2000 * np.ones_like(T)
    Ci, Q, T = np.meshgrid(Ci, Q, T)

    plt.subplot(1, 3, 1)
    p = P.iloc[0].to_dict()
    x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
    A = evaluateFvCB(x, p)
    A = A.reshape(60, 60, 60)
    ci = Ci[0, :, 0]
    a = A[0, :, 0]
    plt.plot(ci, a, "r", linewidth=3)

    plt.xlabel(r"$C_i$ ($\mu$mol mol$^{-1}$)", fontsize=13)
    plt.ylabel(r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    plt.ylim([0, 75])
    plt.grid(True)

    # A vs Q @ T = 25 °C, Ci = 0.7 * 420
    Q = np.linspace(0, 2000, 60)
    T = (25 + 273.15) * np.ones_like(Q)
    Ci = 2000 * np.ones_like(T)
    Ci, Q, T = np.meshgrid(Ci, Q, T)

    plt.subplot(1, 3, 2)
    p = P.iloc[0].to_dict()
    x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
    A = evaluateFvCB(x, p)
    A = A.reshape(60, 60, 60)
    q = Q[:, 0, 0]
    a = A[:, 0, 0]
    plt.plot(q, a, "r", linewidth=3)

    plt.xlabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    plt.ylabel(r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    if(variety==""):
        plt.title(f"{species}", fontsize=15)
    else:
        plt.title(f"{species} var. {variety}", fontsize=15)
    plt.ylim([0, 75])
    plt.grid(True)

    # A vs T @ Q = 2000, Ci = 0.7 * 420
    T = np.linspace(10, 45, 60) + 273.15
    Ci = 2000 * np.ones_like(T)
    Q = 2000 * np.ones_like(T)
    Ci, Q, T = np.meshgrid(Ci, Q, T)

    plt.subplot(1, 3, 3)
    p = P.iloc[0].to_dict()
    x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
    A = evaluateFvCB(x, p)
    A = A.reshape(60, 60, 60)
    t = T[0, 0, :] - 273.15
    a = A[0, 0, :]
    plt.plot(t, a, "r", linewidth=3)

    plt.xlabel(r"$T$ ($^{\circ}$C)", fontsize=13)
    plt.ylabel(r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    plt.ylim([0, 75])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/"+species+variety+"_FvCB_Plot_Responses.png")
    plt.show()

    # 3D Surface Plot
    data = pd.read_csv(compiledDataPath)  # ACi data

    # Create a grid for Ci and T
    Ci = np.linspace(100, 2000, 60)        
    T = np.linspace(273, 40 + 273, 60)        
    Ci, T = np.meshgrid(Ci, T)          
    Q = 2000 * np.ones_like(T)                

    # First subplot: A vs Ci and T at Q = 2000
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    p = P.iloc[0].to_dict()
    x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
    A = evaluateFvCB(x, p)  # Run the FvCB model
    A = A.reshape(Ci.shape)  # Reshape to match the grid

    # Plot modeled surface
    ax1.plot_surface(Ci, T - 273.15, A, cmap='YlGn', edgecolor='none', alpha=0.5, label="FvCB Fit")
    ax1.set_xlabel(r"$C_i$ ($\mu$mol mol$^{-1}$)", fontsize=13)
    ax1.set_ylabel(r"$T$ ($^{\circ}$C)", fontsize=13)
    ax1.set_zlabel(r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax1.view_init(elev=5, azim=-10)

    # Plot measured data
    ax1.scatter(data["Ci"], data["Tleaf"], data["A"], c='r', s=30, label="A-Ci Curves")
    ax1.set_xticks([0,1000,2000])


    # Second subplot: A vs Ci and Q at T = 298.15 K
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    Ci = np.linspace(5, 2000, 60)
    Q = np.linspace(0, 2000, 60)
    Ci, Q = np.meshgrid(Ci, Q)
    T = 298.15 * np.ones_like(Ci)  # Constant temperature at 298.15 K

    x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
    A = evaluateFvCB(x, p)
    A = A.reshape(Ci.shape)

    # Plot modeled surface
    ax2.plot_surface(Ci, Q, A, cmap='YlGn', edgecolor='none', alpha=0.8)#,label="FvCB Fit")
    ax2.set_xlabel(r"$C_i$ ($\mu$mol mol$^{-1}$)", fontsize=13)
    ax2.set_ylabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax2.set_zlabel(r"$A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax2.view_init(elev=5, azim=-10)


    # Plot measured data on modeled surface
    ax2.scatter(data["Ci"], data["Qin"], data["A"], c='r', s=30,label="A-Ci Curves")
    ax2.set_xticks([0,1000,2000])
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if(variety==""):
        plt.suptitle(f"{species}", fontsize=15)
    else:
        plt.suptitle(f"{species} var. {variety}", fontsize=15)
    plt.savefig("results/figures/"+species+variety+"_FvCB_Plot_Surface.png")
    plt.show()

    # 1:1 Comparison of Measured and Modeled Results
    fig, ax = plt.subplots(figsize=(6, 6))
    
    Ci_meas = data["Ci"].to_numpy()
    Q_meas = data["Qin"].to_numpy()
    T_meas = data["Tleaf"].to_numpy() + 273.15
    A_meas = data["A"].to_numpy()  # Measured A

    x_meas = np.column_stack((Ci_meas, Q_meas, T_meas))
    A_model = evaluateFvCB(x_meas, p)
    r2 = computeR2(A_meas, A_model)
    rmse = computeRMSE(A_meas,A_model)
    stats_text = f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.2f}"
    ax.text(0.05, 0.90, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    err = np.abs(A_meas - A_model) / A_meas
    ax.scatter(A_meas, A_model, c=err, cmap='YlGn', label="Data",s=20)
    
    # 1:1 reference line
    min_val = min(A_meas.min(), A_model.min())
    max_val = max(A_meas.max(), A_model.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

    ax.set_xlabel(r"Measured $A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"Modeled $A$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax.legend()
    ax.grid(True)
    
    if(variety==""):
        plt.suptitle(f"{species}", fontsize=15)
    else:
        plt.suptitle(f"{species} var. {variety}", fontsize=15)
    
    plt.savefig("results/figures/"+species+variety+"_FvCB_Plot_R2.png")
    

def saveBMFParametersToFile(species,var,bmf):
    savepath = "results/parameters/"+species+var+"_BMF_Parameters.csv"
    vars = ["species","Em","i0","k","b"]
    vals = [species,bmf.Em[0].item(),bmf.i0[0].item(),bmf.k[0].item(),bmf.b[0].item()]
    outdf = pd.DataFrame([vals],columns=vars)
    outdf.to_csv(savepath,index=False)
    print(f"Parameters saved to: {savepath}")
    return savepath

def plotBMFModelFit(species,variety,parameterPath,dataPath):
    data = pd.read_csv(dataPath,skiprows=[0,2])
    gsw_meas = data["gsw"]
    Q_meas = data["Qamb"]*0.85
    D_meas = data["VPDleaf"]*1000/101.3

    Q = np.linspace(0,2000,50)    
    D = np.linspace(1, 50, 50)   
    Q,D = np.meshgrid(Q, D)

    fig = plt.figure(figsize=[20,9])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    P = pd.read_csv(parameterPath)

    x = np.column_stack((Q.ravel(), D.ravel()))
    p = P.iloc[0].to_dict()
    gsw_modeled = evaluateBMF(x, p)
    gsw_modeled = gsw_modeled.reshape(Q.shape)

    

    ax1.plot_surface(Q, D, gsw_modeled, cmap='YlGn', edgecolor='none', alpha=0.5, label="BMF Fit")
    ax1.set_xlabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax1.set_ylabel(r"$D$ (mmol mol$^{-1}$)", fontsize=13)
    ax1.set_zlabel(r"g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax1.view_init(elev=20, azim=-25)

    ax1.scatter(Q_meas, D_meas, gsw_meas, c='r', s=30, label="Measured gsw")
    ax1.set_xticks([0,1000,2000])
    plt.savefig("results/figures/"+species+variety+"_BMF_Plot_Surface.png")
    plt.show()

    # Plot 1:1 reference line
    x = np.column_stack((Q_meas,D_meas))
    gsw_pred = evaluateBMF(x,p)

    fig, ax = plt.subplots(figsize=(6, 6))
    r2 = computeR2(gsw_meas, gsw_pred)
    rmse = computeRMSE(gsw_meas,gsw_pred)
    stats_text = f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.2f}"
    ax.text(0.05, 0.90, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    err = np.abs(gsw_meas - gsw_pred) / gsw_meas
    ax.scatter(gsw_meas, gsw_pred, c="Green", label="Data",s=20)
    min_val = min(gsw_meas.min(), gsw_pred.min())
    max_val = max(gsw_meas.max(), gsw_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

    ax.set_xlabel(r"Measured g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax.set_ylabel(r"Modeled g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=13)
    ax.legend()
    ax.grid(True)
        
    if(variety==""):
        plt.suptitle(f"{species}", fontsize=15)
    else:
        plt.suptitle(f"{species} var. {variety}", fontsize=15)
        
    plt.savefig("results/figures/"+species+variety+"_BMF_Plot_R2.png")
    plt.show()

def convert_params_to_buffers(dlmodel):
    # Loop through all modules and collect parameters to be modified
    for module in dlmodel.modules():
        # Collect all parameter names and their corresponding values in a list
        params_to_convert = [(name, param) for name, param in module.named_parameters(recurse=False)]

        # Iterate through the collected parameters and modify them
        for name, param in params_to_convert:
            # Detach and clone the parameter
            buffer_param = param.detach().clone()
            # Delete the parameter from the module
            del module._parameters[name]
            # Register the parameter as a buffer in the module
            module.register_buffer(name, buffer_param)
    return dlmodel

def selftest():
    device_test = ['cpu', 'cuda']
    pathlcddfs = 'photorch/data/tests/dfMAGIC043_lr.csv'
    pdlMAGIC043 = pd.read_csv(pathlcddfs)
    lighttypes = [2, 1, 0]
    temptypes = [0, 1, 2]
    onefits = [True, False]
    Rdfits = [True, False]
    KGfits = [True, False]
    lightremoved = False
    count_test = 0
    for idevice in device_test:
        # check if 'cuda' is available
        if idevice == 'cuda':
            if not torch.cuda.is_available():
                idevice = 'cpu'
                print('Cuda is not available, no test will be run on cuda.')
                break
        try:
            print('FvCB testing case: Initialization of Licor data.')
            lcd = fvcb.initLicordata(pdlMAGIC043, preprocess=True, lightresp_id=[118], printout=False)
            lcd.todevice(idevice)
        except:
            raise ValueError('Error in running the FvCB test: Initialization of Licor data failed.')
        for lighttype in lighttypes:
            for temptype in temptypes:
                for onef in onefits:
                    # change the FittingGroup of cureID 5 to 1 or 2
                    if count_test % 2 == 0:
                        pdlMAGIC043.loc[pdlMAGIC043['CurveID'] == 5, 'FittingGroup'] = 2
                    else:
                        pdlMAGIC043.loc[pdlMAGIC043['CurveID'] == 5, 'FittingGroup'] = 1
                    for Rdfit in Rdfits:
                        for KGfit in KGfits:
                            try:
                                print('FvCB testing case:',f'Light type: {lighttype}, Temp type: {temptype}, Onefit: {onef},  fitRd: {Rdfit}, fitKco_Gamma: {KGfit}, Device: {idevice}')
                                if lighttype == 0 and not lightremoved:
                                    lightremoved = True
                                    pdlMAGIC043 = pdlMAGIC043[pdlMAGIC043['CurveID'] != 118]
                                    lcd = fvcb.initLicordata(pdlMAGIC043, preprocess=False, printout=False)
                                    lcd.todevice(idevice)

                                fvcbmMAGIC043 = fvcb.model(lcd, LightResp_type = lighttype, TempResp_type = temptype, onefit = onef, fitgm= True, fitgamma=KGfit, fitKo=KGfit, fitKc=KGfit, fitRd=Rdfit, fitRdratio=~Rdfit, printout=False)
                                resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, maxiteration= 10, minloss=1, recordweightsTF=False, fitcorr=True, printout=False, weakconstiter=5)
                                resultfit.model()
                                # check if all fit parameters are not nan
                            except:
                                raise ValueError('Error in running the FvCB test:',f'Light type: {lighttype}, Temp type: {temptype}, Onefit: {onef},  fitRd: {Rdfit}, fitKco_Gamma: {KGfit}, Device: {idevice}')

    try:
        print('FvCB testing case: Original data without "FittingGroup", "Qin" and "Tleaf".')
        # remove the column "Qin" and "Tleaf"
        pdlMAGIC043 = pdlMAGIC043.drop(columns=['Qin', 'Tleaf','FittingGroup'])
        lcd = fvcb.initLicordata(pdlMAGIC043, preprocess=True, printout=False)
        lcd.todevice(idevice)
        fvcbmMAGIC043 = fvcb.model(lcd, LightResp_type=0, TempResp_type=0, printout=False)
        resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, maxiteration= 10, minloss=1, recordweightsTF=False, fitcorr=False, printout=False)
        resultfit.model()
    except:
        raise ValueError('Error in running the FvCB test: Original data without "FittingGroup", "Qin" and "Tleaf".')

    try:
        print('FvCB testing case: Reset parameters and record weights.')
        allparams = fvcb.allparameters()
        allparams.alphaG = torch.tensor([0.1]).to(idevice)
        fvcbmMAGIC043 = fvcb.model(lcd, LightResp_type=0, TempResp_type=0, printout=False, allparams=allparams)
        resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, maxiteration=10, minloss=1, recordweightsTF=True, fitcorr=False, printout=False)
        resultfit.model()
    except:
        raise ValueError('Error in running the FvCB test: Reset parameters and record weights.')

    stomatallabels = ['BMF','BWB','MED']
    for stomataltype in stomatallabels:
        try:
            datasc = pd.read_csv('photorch/data/tests/steadystate_stomatalconductance.csv')
            scd = stomatal.initscdata(datasc, printout=False)
        except:
            raise ValueError('Error in running the stomatal conductance test: Initialization of stomatal data failed.')
        if stomataltype == 'BMF':
            try:
                print('Stomatal conductance testing case: "BMF"')
                scm = stomatal.BMF(scd)
                resultfit = stomatal.fit(scm, learnrate=0.5, maxiteration=20, printout=False)
                resultfit.model()
            except:
                raise ValueError('Error in running the stomatal conductance test: "BMF"')
        elif stomataltype == 'BWB':
            try:
                print('Stomatal conductance testing case: "BWB"')
                scm = stomatal.BWB(scd)
                resultfit = stomatal.fit(scm, learnrate=0.5, maxiteration=20, printout=False)
                resultfit.model()
            except:
                raise ValueError('Error in running the stomatal conductance test: "BWB"')
        elif stomataltype == 'MED':
            try:
                print('Stomatal conductance testing case: "MED"')
                scm = stomatal.MED(scd)
                resultfit = stomatal.fit(scm, learnrate=0.5, maxiteration=20, printout=False)
                resultfit.model()
            except:
                raise ValueError('Error in running the stomatal conductance test: "MED"')

    print('All FvCB tests passed!')
    print('All stomatal conductance tests passed!')