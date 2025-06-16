# Process ACi Data and Verify the Printed Curves are Those Desired to Fit

################# User Settings #################

fitting_group_folder_path = "photorch/data/fvcb/curves/iceberg"
species_to_fit = "Iceberg"
species_variety = "Calmar"

#################################################
from photorch import *
compiledDataPath = util.compileACiFiles(fitting_group_folder_path)

# Fit Compiled Data to FvCB Model using PhoTorch, Save Parameters, and Plot Results

######### User Settings ##########
LightResponseType = 2
TemperatureResponseType = 2
Fitgm = False
FitGamma = False
FitKc = False
FitKo = False
saveParameters = True
plotResultingFit = True
#### Advanced Hyper Parameters ####
learningRate = 0.08
iterations = 10000
###################################

import torch
import pandas as pd
df = pd.read_csv(fitting_group_folder_path+"/curves.csv")
lcd = fvcb.initLicordata(df,preprocess=True)
device_fit = 'cpu'
lcd.todevice(torch.device(device_fit))

if(species_variety==""):
    print(f"Fitting {species_to_fit}")
else:
    print(f"Fitting {species_to_fit} var. {species_variety}")

fvcbm = fvcb.model(lcd, LightResp_type=LightResponseType, TempResp_type=TemperatureResponseType, onefit=True, fitgamma = FitGamma, fitKc=FitKc, fitKo=FitKo, fitgm=Fitgm)
fitresult = fvcb.fit(fvcbm, learn_rate= learningRate, maxiteration = iterations, minloss= 1, recordweightsTF=False)
fvcbm = fitresult.model

util.printFvCBParameters(fvcbm,LightResponseType,TemperatureResponseType,Fitgm,FitGamma,FitKc,FitKo)
if(saveParameters):
    parameterPath = util.saveFvCBParametersToFile(species_to_fit,species_variety,fvcbm,LightResponseType,TemperatureResponseType,Fitgm,FitGamma,FitKc,FitKo)
if(plotResultingFit):
    util.plotFvCBModelFit(species_to_fit,species_variety,parameterPath,compiledDataPath)