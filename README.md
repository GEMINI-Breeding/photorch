# PhoTorch

PhoTorch is a robust and generalized photosynthesis biochemical model fitting package based on PyTorch.
Read more about PhoTorch in our paper: https://arxiv.org/abs/2501.15484.
Currently, the package includes the Farquhar, von Caemmerer, and Berry (FvCB) model. The series of stomatal conductance models and the PROSPECT-X model are under development.
## Installation of dependencies
```bash
pip install torch
pip install numpy
pip install scipy
pip install pandas
```
## 1. FvCB model usage
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
dftest = pd.read_csv('exampledata/dfMAGIC043_lr.csv')
# remove the rows with negative A values
dftest = dftest[dftest['A'] > 0]
```
### Initialize the data
```bash
# specify the list of light response curve IDs, if no light response curve, input "lightresp_id = None"
lcd = fitaci.initD.initLicordata(dftest, preprocess=True, lightresp_id = [118])
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
fvcbm = fitaci.initM.FvCB(lcd, LightResp_type = 2, TempResp_type = 2, onefit = False)
```
### More fitting options
fitRd: option to fit $R_{d25}$, default is True. If set to False, $R_{d}$ will be fixed to 1% of $V_{cmax}$.

fitRdratio: option to fit $R_{d}$-to- $V_{cmax}$ ratio, default is False, the range is 0.01 to 0.02. 

fitag: option to fit $\alpha_g$, default is False, the range is 0 to 1.

fitKc: option to fit $k_{25}$, default is False.

fitKo: option to fit $k_{25}$, default is False.

fitgamma: option to fit $\Gamma^*_{25}$, default is False.

fitgm: option to fit $g_m$, default is False.
```bash
fvcbm = fitaci.initM.FvCB(lcd, LightResp_type = 0, TempResp_type = 1, onefit = False, fitRd = True, fitRdratio = False, fitag = False, fitgm= False, fitgamma=False, fitKo=False, fitKc=False, allparams=allparams)
```
### Specify default fixed or learnable parameters
```bash
allparams = fitaci.initM.allparameters()
allparams.dHa_Vcmax = torch.tensor(40.0)
fvcbm = fitaci.initM.FvCB(lcd, LightResp_type = 0, TempResp_type = 1, onefit = False, fitag = False, fitgm= False, fitgamma=False, fitKo=False, fitKc=False, allparams=allparams)
```
### Fit A/Ci curves
```bash
fitresult = fitaci.run(fvcbm, learn_rate= 0.08, device=device_fit, maxiteration = 20000, minloss= 1, recordweightsTF=False)
fvcbm = fitresult.model
```
### Get fitted parameters by ID
The main parameters are stored in the 'fvbm'. The temperature response parameters are in 'fvcbm.TempResponse', just like the light response parameters.
```bash
id_index = 0
id = int(lcd.IDs[id_index]) # target curve ID
fg_index =  int(lcd.FGs_idx[id_index]) # index of the corresponding fitting group
if not fvcbm.onefit:
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
Ac_id = Ac[indices_id]
```
### Get the (preprocessed) photosynthesis data by ID
```bash
A_id_mea, Ci_id, Q_id, Tlf_id = lcd.getDatabyID(lcd.IDs[id_index])
```
***

## 2. Stomatal conductance model usage
The stomatal conductance model is under development.
```bash
import stomatal
```
### Initialize stomatal conductance models
One stomatal conductance model is currently available: Buckley Mott Farquhar (BMF). Other three Ball Berry Leuning (BBL), Medlyn (MED), and Ball Woodrow Berry (BWB) are under development.
More details about these four models can be found at: https://baileylab.ucdavis.edu/software/helios/_stomatal_doc.html.

### Initialize the stomatal conductance data
The data to be loaded should be:

| CurveID | gsw | VPDleaf_mmolmol-1 | A  | Qin   | Tleaf | RH   |
|---------|-----|-------------------|----|-------|-------|------|
| 0       | 0.34| 11.32             | 55 | 2000  | 21.81 | 21.0 |
| 0       | 0.34| 18.33             | 55 | 2000  | 22.71 | 30.02|
| 0       | 0.35| 29.57             | 51 | 2000  | 20.02 | 38.01|
| 0       | 0.38| 15.4              | 54 | 2000  | 22.6  | 26.99|
| 0       | 0.32| 15.44             | 54 | 1200  | 19.97 | 27.0 |
| 1       | 0.23| 29.03             | 49 | 2000  | 17.93 | 35.92|
| 1       | 0.29| 20.51             | 50 | 2000  | 20.51 | 29.96|
| 1       | 0.28| 11.77             | 49 | 2000  | 18.61 | 19.99|

```bash
datasc = pd.read_csv('exampledata/steadystate_stomatalconductance.csv')
scd = stomatal.initscdata(datasc)
```
### Initialize the BMF model and fit the parameters Emerson effect (Em), quantum yield of electron transport (i0), curvature factor (k), and intercept (b).
```bash
scm = stomatal.BMF(scd)
scm = stomatal.run(scm, learnrate = 0.5, maxiteration =20000)
```
### Get the fitted and measured stomatal conductance
```bash
gsw = scm()
gsw_mea = scd.gsw
```
### Get the fitted stomatal conductance by ID
```bash
id_index = 0
id = scd.IDs[id_index]
indices_id = scd.getIndicesbyID(id)
gsw_id = gsw[indices_id]
```
### Get the fitted parameters by ID
```bash
id_index = 0
id = int(scd.IDs[id_index])
Em_id = scm.Em[id_index]
i0_id = scm.i0[id_index]
k_id = scm.k[id_index]
b_id = scm.b[id_index]
```
  
## 3. PROSPECT-X model usage
The PROSPECT-X model is under development.
```bash
import prospectmodels
```
