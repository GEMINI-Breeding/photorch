import pandas as pd
import fitaci as fitACi
import torch

def run():
    device_test = ['cpu', 'cuda']
    pathlcddfs = 'dfMAGIC043_lr.csv'
    pdlMAGIC043 = pd.read_csv(pathlcddfs)
    lighttypes = [2, 1, 0]
    temptypes = [0, 1, 2]
    onefits = [True, False]
    Rdfits = [True, False]
    KGfits = [True, False]

    for idevice in device_test:
        # check if 'cuda' is available
        if idevice == 'cuda':
            if not torch.cuda.is_available():
                break
        lcd = fitACi.initD.initLicordata(pdlMAGIC043, preprocess=True, lightresp_id=[118], printout=False)
        lcd.todevice(idevice)
        for lighttype in lighttypes:
            for temptype in temptypes:
                for onef in onefits:
                    for Rdfit in Rdfits:
                        for KGfit in KGfits:
                            try:
                                print('Testing case:',f'Light type: {lighttype}, Temp type: {temptype}, Onefit: {onef},  fitRd: {Rdfit}, fitKco_Gamma: {KGfit}, Device: {idevice}')
                                if lighttypes == 0:
                                    pdlMAGIC043 = pdlMAGIC043[pdlMAGIC043['CurveID'] != 118]
                                    lcd = fitACi.initD.initLicordata(pdlMAGIC043, preprocess=True, printout=False)
                                    lcd.todevice(idevice)

                                fvcbmMAGIC043 = fitACi.initM.FvCB(lcd, LightResp_type = lighttype, TempResp_type = temptype, onefit = onef, fitgm= True, fitgamma=KGfit, fitKo=KGfit, fitKc=KGfit,fitRd=Rdfit, fitRdratio=~Rdfit,printout=False)
                                resultfit = fitACi.run(fvcbmMAGIC043, learn_rate=0.8, device=idevice, maxiteration=5, minloss=1, recordweightsTF=False, fitcorr=True, printout=False)
                                resultfit.model()
                            except:
                                raise ValueError('Error in running the test:',f'Light type: {lighttype}, Temp type: {temptype}, Onefit: {onef},  fitRd: {Rdfit}, fitKco_Gamma: {KGfit}, Device: {idevice}')

    print('All tests passed!')

