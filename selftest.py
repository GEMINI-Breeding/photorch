import pandas as pd
import torch
import stomatal
import fvcb
def run():
    device_test = ['cpu', 'cuda']
    pathlcddfs = 'exampledata/dfMAGIC043_lr.csv'
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
                                resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, device=idevice, maxiteration= 10, minloss=1, recordweightsTF=False, fitcorr=True, printout=False,weakconstiter=5)
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
        resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, device=idevice, maxiteration= 10, minloss=1, recordweightsTF=False, fitcorr=False, printout=False)
        resultfit.model()
    except:
        raise ValueError('Error in running the FvCB test: Original data without "FittingGroup", "Qin" and "Tleaf".')

    try:
        print('FvCB testing case: Reset parameters and record weights.')
        allparams = fvcb.allparameters()
        allparams.alphaG = torch.tensor([0.1]).to(idevice)
        fvcbmMAGIC043 = fvcb.model(lcd, LightResp_type=0, TempResp_type=0, printout=False, allparams=allparams)
        resultfit = fvcb.fit(fvcbmMAGIC043, learn_rate=0.8, device=idevice, maxiteration=10, minloss=1, recordweightsTF=True, fitcorr=False, printout=False)
        resultfit.model()
    except:
        raise ValueError('Error in running the FvCB test: Reset parameters and record weights.')

    stomatallabels = ['BMF','BWB','MED']
    for stomataltype in stomatallabels:
        try:
            datasc = pd.read_csv('exampledata/steadystate_stomatalconductance.csv')
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

