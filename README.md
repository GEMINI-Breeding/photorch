# PhoTorch

PhoTorch is a robust and generalized photosynthesis biochemical model fitting package based on PyTorch.

## Installation of dependencies
```bash
pip install torch
pip install numpy
pip install scipy
pip install pandas
```
## Usage
After installing the dependencies, download the package and import it into your Python script.

```bash
import fitaci
import pandas as pd
import torch
```
### Load data
Load the example CSV file. Then, specify the ID of the light response curve. If there is no light response curve in the dataset, ignore it.
The loaded data frame should have columns with titles 'CurveID', 'FittingGroup', 'Ci', 'A', 'Qin', and 'Tleaf'. Each A/Ci curve should have a unique 'CurveID'.
If no 'Qin' and 'Tleaf' are available, it will be automatically set to 2000 and 25, respectively.

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
# specify the list of light response curve IDs, if no light response curve, input "lightresp_id = None"
lcd = fitACi.initD.initLicordata(dftest, preprocess=True, lightresp_id = [118])
```
### Define the device
'cuda' means an NVIDIA GPU will be used. If you want to use the CPU, set 'device_fit' to 'cpu'.
```bash
device_fit = 'cpu'
lcd.todevice(torch.device(device_fit)) # if device is cuda, then execute this line
```
### Initialize FvCB model
If 'onefit' is set to 'True', all curves in a fitting group will share the same set of Vcmax25, Jmax25, TPU25, and Rd25.
Otherwise, each curve will have its own set of these four main parameters but share the same light and temperature response parameters for the fitting group.

If no light response curve is specified, set 'LightResp_type' to 0.

LightResp_type 0: J is equal to Jmax.

LightResp_type 1: using equation $J = \frac{\alpha Q J_{max}}{\alpha Q + J_{max}}$ and fitting $\alpha$.

LightResp_type 2: using equation $J = \frac{\alpha Q + J_{max} - \sqrt{(\alpha Q + J_{max})^2 - 4 \theta \alpha Q J_{max}}}{2 \theta}$ and fitting $\alpha$ and $\theta$.

TempResp_type 0: Vcmax, Jmax, TPU, and Rd are equal to the Vcmax25, Jmax25, TPU25, and Rd25, respectively.

TempResp_type 1: using equation $k = k_{25} \exp{\left[\frac{\Delta{H_a}}{R}\left(\frac{1}{298}-\frac{1}{T_{leaf}}\right)\right]}$ and fitting $\Delta{H_a}$ for Vcmax, Jmax, and TPU.

TempResp_type 2: using equation $k = k_{25} \exp\left[\frac{\Delta H_a}{R} \left(\frac{1}{298}-\frac{1}{T_{leaf}}\right)\right]  \frac{f\left(298\right)}{f\left(T_{leaf}\right)}$, where $f(T) = 1+\exp \left[\frac{\Delta H_d}{R}\left(\frac{1}{T_{opt}}-\frac{1}{T} \right)-\ln \left(\frac{\Delta H_d}{\Delta H_a}-1 \right) \right]$, and fitting $\Delta{H_a}$ and $T_{opt}$ for Vcmax, Jmax, and TPU.

```bash
# initialize the model
fvcbm = fitACi.initM.FvCB(lcd, LightResp_type = 2, TempResp_type = 2, onefit = False)
```
### (Alternatively) Specify default fixed or learnable parameters, and set whether to fit Kc25, Ko25, gamma25, and gm (all defaults are False)
```bash
allparamas = fitACi.initM.allparameters()
allparamas.dHa_Vcmax = torch.tensor(40.0)
fvcbm = fitACi.initM.FvCB(lcd, LightResp_type = 0, TempResp_type = 1, onefit = False, fitgm= False, fitgamma=True, fitKo=False, fitKc=True, allparams=allparamas)
```
### Fit A/Ci curves
```bash
fitresult = fitACi.run(fvcbm, learn_rate= 0.08, device=device_fit, maxiteration = 20000, minloss= 1, recordweightsTF=False)
fvcbm = fitresult.model
```
### Get fitted parameters by ID
The main parameters are stored in the 'fvbm'. The temperature response parameters are in 'fvcbm.TempResponse', just like the light response parameters.
```bash
id_index = 0
id = int(lcd.IDs[id_index]) # target curve ID
fg_index =  int(lcd.FGs[id_index]) # index of the corresponding fitting group
if fvcbm.onefit:
    Vcmax25_id = fvcbm.Vcmax25[id_index]
    Jmax25_id = fvcbm.Jmax25[id_index]
else:
    Vcmax25_id = fvcbm.Vcmax25[fg_index]
    Jmax25_id = fvcbm.Jmax25[fg_index]

dHa_Vcmax_id = fvcbm.TempResponse.dHa_Vcmax[fg_index]
alpha_id = fvcbm.LightResponse.alpha[fg_index]
```
### Get fitted A/Ci curves
```bash
A, Ac, Aj, Ap = fvcbm()
```

### Get fitted A/Ci curves by ID
```bash
id_index = 0
id = lcd.IDs[id_index]
indices_id = lcd.getIndicesbyID(id)
A_id = A[indices_id]
```
### Get the (preprocessed) photosynthesis data by ID
```bash
A_id_mea, Ci_id, Q_id, Tlf_id = lcd.getDatabyID(lcd.IDs[id_index])
```
