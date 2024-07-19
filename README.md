# PhoTorch

PhoTorch is a robust and generalized photosynthesis biochemical model fitting package using a PyTorch-based optimizer.

## Installation
```bash
pip install pytorch
pip install numpy
pip install scipy
pip install pandas
```
## Usage
```bash
import fitaci
import pandas as pd
```
### Load data
Load the example CSV file. Then, specify the ID of the light response curve. If there is no light response curve in the dataset, ignore it.
The loaded data frame should have columns with titles 'CurveID', 'FittingGroup', 'Ci', 'A', 'Qin', and 'Tleaf'. Each A/Ci curve should have a unique 'CurveID'. The data in the 'CurveID' and 'FittingGroup' columns should be integers.

The data to be loaded should be:

| CurveID | FittingGroup | Ci  | A  | Qin  | Tleaf |
|---------|--------------|-----|----|------|-------|
| 1       | 1            | 200 | 20 | 2000 | 25    |
| 1       | 1            | 400 | 30 | 2000 | 25    |
| 1       | 1            | 600 | 40 | 2000 | 25    |
| 2       | 1            | 200 | 25 | 2000 | 30    |
| 2       | 1            | 400 | 35 | 2000 | 30    |
| 2       | 1            | 700 | 55 | 2000 | 30    |

```bash
dftest = pd.read_csv('dfMAGIC043_lr.csv')
# initialize the data, and preprocess the data
lcd = fitACi.initD.initLicordata(dftest, preprocess=True)
# specify the ID of the light response curve, if no light response curve, ignore this line
id_lresp= 118
lcd.setLightRespID(id_lresp)
```
### Define the device
```bash
device_fit = 'cpu'
lcd.todevice(torch.device(device_fit)) # if device is cuda, then execute this line
```
### Initialize FvCB model
If 'onefit' is set to 'True', all curves in a fitting group will share the same set of Vcmax25, Jmax25, TPU25, and Rd25 (or Vcmax etc. if TempResp_type is 0).
Otherwise, each curve will have its own set of these four main parameters but share the same light and temperature response parameters for the fitting group.

If no light response curve is specified, set 'LightResp_type' to 0.
```bash
# initialize the model
fvcbm = fitACi.initM.FvCB(lcd, LightResp_type = 2, TempResp_type = 2, onefit = False, fitgm=False)
```
### Fit A/Ci curves
```bash
fitresult = fitACi.run(fvcbm, learn_rate= 0.08, device=device_fit, maxiteration = 20000, minloss= 1, recordweightsTF=False)
fvcbm = fitresult.model
```
### Get fitted parameters
The main parameters are stored in the 'fvbm'. The temperature response parameters are in 'fvcbm.TempResponse', just like the light response parameters.
```bash
id_index = 0
id = int(lcd.IDs[id_index])
fg_index =  int(lcd.FGs[id_index])
if fvcbm.onefit:
    Vcmax25_id = fvcbmMAGIC043.Vcmax25[id_index]
    Jmax25_id = fvcbmMAGIC043.Jmax25[id_index]
else:
    Vcmax25_id = fvcbmMAGIC043.Vcmax25[fg_index]
    Jmax25_id = fvcbmMAGIC043.Jmax25[fg_index]

dHa_Vcmax_id = fvcbmMAGIC043.TempResponse.dHa_Vcmax[fg_index]
alpha_id = fvcbmMAGIC043.LightResponse.alpha[fg_index]
```
### Get fitted A/Ci curves
```bash
A, Ac, Aj, Ap = fvcbm()
```

### Get fitted A/Ci curves by ID
```bash
A, Ac, Aj, Ap = fvcbm()
id_index = 0
id = lcd.IDs[id_index]
indices_id = lcd.getIndicesbyID(id)
A_id = A[indices_id]
```
### Get the original (preprocessed) photosynthesis data by ID
```bash
A_id_mea, Ci_id, Q_id, Tlf_id = lcd.getDatabyID(lcd.IDs[i])
```
