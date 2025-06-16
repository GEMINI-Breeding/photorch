# Process Stomatal Data and Observe the Model Fit
################# User Settings #################

fitting_group_folder_path = "photorch/data/stomatal/survey/iceberg/Iceberg_poro"
species_to_fit = "Iceberg"
species_variety = "Calmar"

#################################################

import pandas as pd
from photorch import *

# Stomatal model fitting
data = pd.read_csv(fitting_group_folder_path+'.csv',skiprows=[0,2])
data["CurveID"] = 0

scd = stomatal.initscdata(data)
scm = stomatal.BMF(scd)
scm = stomatal.fit(scm, learnrate = 0.5, maxiteration = 20000)
gsw = scm.model()
gsw_pred = gsw.detach().numpy()
gsw_meas = scd.gsw.numpy()

parameterPath = util.saveBMFParametersToFile(species_to_fit,species_variety,scm.model)
util.plotBMFModelFit(species_to_fit,species_variety,parameterPath,fitting_group_folder_path+".csv")