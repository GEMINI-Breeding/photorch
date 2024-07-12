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
```bash
dftest = pd.read_csv('dfMAGIC043_lr.csv')
lcd = fitACi.initD.initLicordata(dftest, preprocess=True)
id_lresp= 118
lcd.setLightRespID(id_lresp)
```
### Define the device
```bash
device_fit = 'cpu'
lcd.todevice(torch.device(device_fit)) # if device is cuda, then execute this line
```
### Initialize FvCB model
```bash
fvcbm = fitACi.initM.FvCB(lcd, LightResp_type = 2, TempResp_type = 2, onefit = False, fitgm=False)
```
### Fit A/Ci curves
```bash
fvcbm,recordweights = fitACi.run(fvcbm,learn_rate= 0.08, device=device_fit, maxiteration = 20500, minloss= 1,recordweightsTF=True)
```
### Get the fitted parameters
The main parameters are stored in the 'fvbm'. The temperature response parameters are in 'fvcbm.TempResponse', just like the light response parameters.
```bash
Vcmax25 = fvcbm.Vcmax25
dHa_Vcmax = fvcbm.TempResponse.dHa_Vcmax
alpha = fvcbm.LightResponse.dHa_Vcmax
```
### Get fitted A/Ci curves
```bash
A, Ac, Aj, Ap, Gamma = fvcbm()
```

### Get fitted A/Ci curves by ID
```bash
A, Ac, Aj, Ap, Gamma = fvcbm()
id_index = 0
id = lcd.IDs[id_index]
indices_id = lcd.getIndicesbyID(id)
A_id = A[indices_id]
```
### Get the original (preprocessed) photosynthesis data by ID
```bash
A_id_mea, Ci_id, Q_id, Tlf_id = lcd.getDatabyID(lcd.IDs[i])
```
