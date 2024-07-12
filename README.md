# PhoTorch

PhoTorch is a a robust and generalized photosynthesis biochemical model fitting package using a PyTorch-based optimizer.

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
```bash
dftest = pd.read_csv('dfMAGIC043_lr.csv')
lcd = fitACi.initD.initLicordata(dftest, preprocess=True)
lcd.setLightRespID(118)
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
