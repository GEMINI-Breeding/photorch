import torch.nn as nn
import torch

class initTRparameters(nn.Module):
    def __init__(self):
        super(initTRparameters, self).__init__()
        # constants

        self.R = torch.tensor(0.0083144598)
        self.kelvin = torch.tensor(273.15)
        self.Troom = torch.tensor(25.0) + self.kelvin

        self.c_Vcmax = torch.tensor(26.35) #Improved temperature response functions for models of Rubisco-limited photosynthesis
        self.c_Jmax = torch.tensor(17.71)   #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.c_TPU = torch.tensor(21.46)  #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.c_Rd = torch.tensor(18.72) #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.c_Gamma = torch.tensor(19.02) #Improved temperature response functions for models of Rubisco-limited photosynthesis
        self.c_Kc = torch.tensor(38.05)  #Improved temperature response functions for models of Rubisco-limited photosynthesis
        self.c_Ko = torch.tensor(20.30)  #Improved temperature response functions for models of Rubisco-limited photosynthesis
        # self.c_gm = torch.tensor(20.01) #Fitting photosynthetic carbon dioxide response curves for C3 leaves

        self.dHa_Vcmax = torch.tensor(65.33) #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.dHa_Jmax = torch.tensor(43.9)   #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.dHa_TPU = torch.tensor(53.1)  #Modelling photosynthesis of cotton grown in elevated CO2
        self.dHa_Rd = torch.tensor(46.39) #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        self.dHa_Gamma = torch.tensor(37.83)  #Improved temperature response functions for models of Rubisco-limited photosynthesis
        self.dHa_Kc = torch.tensor(79.43)  #Improved temperature response functions for models of Rubisco-limited photosynthesis
        self.dHa_Ko = torch.tensor(36.38)  #Improved temperature response functions for models of Rubisco-limited photosynthesis
        # self.dHa_gm = torch.tensor(49.6) #Fitting photosynthetic carbon dioxide response curves for C3 leaves

        self.dHd_Vcmax = torch.tensor(200.0) #Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data
        self.dHd_Jmax = torch.tensor(200.0) #Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data
        self.dHd_TPU = torch.tensor(201.8)  #Fitting photosynthetic carbon dioxide response curves for C3 leaves #Modelling photosynthesis of cotton grown in elevated CO2
        # self.dHd_gm = torch.tensor(437.4) #Fitting photosynthetic carbon dioxide response curves for C3 leaves

        self.Topt_Vcmax = torch.tensor(38.0) + self.kelvin #Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data
        self.Topt_Jmax = torch.tensor(38.0) + self.kelvin  #Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data
        self.dS_TPU = torch.tensor(0.65)  #Fitting photosynthetic carbon dioxide response curves for C3 leaves # Modelling photosynthesis of cotton grown in elevated CO2
        self.Topt_TPU = self.dHd_TPU/(self.dS_TPU-self.R * torch.log(self.dHa_TPU/(self.dHd_TPU-self.dHa_TPU)))
        # self.dS_gm = torch.tensor(1.4) #Fitting photosynthetic carbon dioxide response curves for C3 leaves
        # self.Topt_gm = self.dHd_gm/(self.dS_gm-self.R * torch.log(self.dHa_gm/(self.dHd_gm-self.dHa_gm)))

        #different species have different parameters
    # Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data


class LightResponse(nn.Module):
    def __init__(self, lcd, lr_type: int = 0):
        super(LightResponse, self).__init__()
        self.Q = lcd.Q
        self.type = lr_type
        self.FGs = lcd.FGs
        self.lengths = lcd.lengths
        self.num_FGs = lcd.num_FGs

        if self.type == 0:
            print('Light response type 0: No light response.')
            self.alpha = torch.tensor(0.5).to(lcd.device)
            self.Q_alpha = self.Q * self.alpha
            self.getJ = self.Function0

        elif self.type == 1:
            print('Light response type 1: alpha will be fitted.')
            self.alpha = nn.Parameter(torch.ones(self.num_FGs) * 0.5)
            self.alpha = nn.Parameter(self.alpha)
            self.getJ = self.Function1

        elif self.type == 2:
            print('Light response type 2: alpha and theta will be fitted.')
            self.alpha = nn.Parameter(torch.ones(self.num_FGs) * 0.5)
            self.theta = nn.Parameter(torch.ones(self.num_FGs) * 0.7)
            self.getJ = self.Function2
        else:
            raise ValueError('LightResponse type should be 0 (no light response), 1 (alhpa), or 2 (alpha and theta)')

    def Function0(self, Jmax):
        J = Jmax * self.Q_alpha / (self.Q_alpha + Jmax)
        return J
    def Function1(self, Jmax):
        if self.num_FGs > 1:
            alpha = torch.repeat_interleave(self.alpha[self.FGs], self.lengths, dim=0)
        else:
            alpha = self.alpha
        J = Jmax * self.Q * alpha / (self.Q * alpha + Jmax)
        return J

    def Function2(self, Jmax):
        theta = torch.clamp(self.theta, min=0.0001)
        if self.num_FGs > 1:
            alpha = torch.repeat_interleave(self.alpha[self.FGs], self.lengths, dim=0)
            theta = torch.repeat_interleave(theta[self.FGs], self.lengths, dim=0)
        else:
            alpha = self.alpha
        alphaQ_J = torch.pow(alpha * self.Q + Jmax, 2) - 4 * alpha * self.Q * Jmax * theta
        alphaQ_J = torch.clamp(alphaQ_J, min=0)
        J = alpha * self.Q + Jmax - torch.sqrt(alphaQ_J)
        J = J / (2 * theta)
        return J


class TemperatureResponse(nn.Module):
    def __init__(self, lcd, TR_type: int = 0):
        super(TemperatureResponse, self).__init__()
        self.Tleaf = lcd.Tleaf
        self.type = TR_type
        self.FGs = lcd.FGs
        self.lengths = lcd.lengths
        self.num_FGs = lcd.num_FGs
        device = lcd.device
        self.TRparam = initTRparameters()
        onetensor = torch.ones(1).to(device)
        self.R_Tleaf = self.TRparam.R * self.Tleaf
        self.R_kelvin = self.TRparam.R * self.TRparam.Troom
        self.R_kelvin = self.R_kelvin.to(device)
        if self.type == 0:
            self.getVcmax = lambda x: x
            self.getJmax = lambda x: x
            self.getRd = lambda x: x
            self.getTPU = lambda x: x
            print('Temperature response type 0: No temperature response.')

        elif self.type == 1:
            # repeat dHa_Rd with self.num_FGs repeated
            self.dHa_Rd = self.TRparam.dHa_Rd.repeat(self.num_FGs).to(device)
            self.Rd_tw = self.tempresp_fun1(onetensor, self.dHa_Rd)
            # initial paramters with self.num_FGs repeated
            self.dHa_Vcmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_Vcmax)
            self.dHa_Jmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_Jmax)
            self.dHa_TPU = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_TPU)
            self.getVcmax = self.getVcmaxF1
            self.getJmax = self.getJmaxF1
            self.getTPU = self.getTPUF1
            self.getRd = self.getRdF0
            print('Temperature response type 1: dHa_Vcmax, dHa_Jmax, dHa_TPU will be fitted.')

        elif self.type == 2:
            self.dHa_Rd = self.TRparam.dHa_Rd.repeat(self.num_FGs).to(device)
            self.Rd_tw = self.tempresp_fun1(onetensor, self.dHa_Rd)
            self.dHa_Vcmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_Vcmax)
            self.dHa_Jmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_Jmax)
            self.dHa_TPU = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.dHa_TPU)
            self.Topt_Vcmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.Topt_Vcmax)
            self.Topt_Jmax = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.Topt_Jmax)
            self.Topt_TPU = nn.Parameter(torch.ones(self.num_FGs) * self.TRparam.Topt_TPU)
            self.getVcmax = self.getVcmaxF2
            self.getJmax = self.getJmaxF2
            self.getTPU = self.getTPUF2
            self.getRd = self.getRdF0
            self.dHd_Vcmax = self.TRparam.dHd_Vcmax
            self.dHd_Jmax = self.TRparam.dHd_Jmax
            self.dHd_TPU = self.TRparam.dHd_TPU
            self.dHd_R_Vcmax = self.dHd_Vcmax / self.TRparam.R
            self.dHd_R_Jmax = self.dHd_Jmax / self.TRparam.R
            self.dHd_R_TPU = self.dHd_TPU / self.TRparam.R
            self.rec_Troom = 1 / self.TRparam.Troom
            self.rec_Tleaf = 1 / self.Tleaf

            print('Temperature response type 2: dHa_Jmax, dHa_TPU, Topt_Vcmax, Topt_Jmax, Topt_TPU will be fitted.')
        else:
            raise ValueError('TemperatureResponse type should be 0, 1 or 2')

        self.dHa_Gamma = self.TRparam.dHa_Gamma.repeat(self.num_FGs).to(device)
        self.dHa_Kc = self.TRparam.dHa_Kc.repeat(self.num_FGs).to(device)
        self.dHa_Ko = self.TRparam.dHa_Ko.repeat(self.num_FGs).to(device)

        self.Gamma_tw = self.tempresp_fun1(onetensor, self.dHa_Gamma)
        self.Kc_tw = self.tempresp_fun1(onetensor,  self.dHa_Kc)
        self.Ko_tw = self.tempresp_fun1(onetensor,  self.dHa_Ko)

        self.geGamma = self.getGammF0
        self.getKc = self.getKcF0
        self.getKo = self.getKoF0

    def tempresp_fun1(self, k25, dHa):
        if self.num_FGs > 1:
            dHa = torch.repeat_interleave(dHa[self.FGs], self.lengths, dim=0)
        k = k25 * torch.exp(dHa /self.R_kelvin - dHa / self.R_Tleaf)
        return k

    def tempresp_fun2(self, k25, dHa, dHd, Topt, dHd_R):
        if self.num_FGs > 1:
            dHa = torch.repeat_interleave(dHa[self.FGs], self.lengths, dim=0)
            Topt = torch.repeat_interleave(Topt[self.FGs], self.lengths, dim=0)
        k_1 = self.tempresp_fun1(k25, dHa)
        dHd_dHa = dHd / dHa
        dHd_dHa = torch.clamp(dHd_dHa, min=1.0001)
        log_dHd_dHa = torch.log(dHd_dHa - 1)
        rec_Top = 1/Topt
        k = k_1 * (1 + torch.exp(dHd_R * (rec_Top - self.rec_Troom) - log_dHd_dHa)) / (1 + torch.exp(dHd_R * (rec_Top - self.rec_Tleaf) - log_dHd_dHa))
        return k

    def getRdF0(self, Rd25):
        Rd = Rd25 * self.Rd_tw
        return Rd

    def getGammF0(self, Gamma25):
        Gamma = Gamma25 * self.Gamma_tw
        return Gamma

    def getKcF0(self, Kc25):
        Kc = Kc25 * self.Kc_tw
        return Kc

    def getKoF0(self, Ko25):
        Ko = Ko25 * self.Ko_tw
        return Ko

    def getVcmaxF1(self, Vcmax25):
        Vcmax = self.tempresp_fun1(Vcmax25, self.dHa_Vcmax)
        return Vcmax

    def getJmaxF1(self, Jmax25):
        Jmax = self.tempresp_fun1(Jmax25, self.dHa_Jmax)
        return Jmax

    def getTPUF1(self, TPU25):
        TPU = self.tempresp_fun1(TPU25, self.dHa_TPU)
        return TPU

    def getVcmaxF2(self, Vcmax_o):
        Vcmax = self.tempresp_fun2(Vcmax_o, self.dHa_Vcmax, self.dHd_Vcmax, self.Topt_Vcmax, self.dHd_R_Vcmax)
        return Vcmax

    def getJmaxF2(self, Jmax_o):
        Jmax = self.tempresp_fun2(Jmax_o, self.dHa_Jmax, self.dHd_Jmax, self.Topt_Jmax, self.dHd_R_Jmax)
        return Jmax

    def getTPUF2(self, TPU_o):
        TPU = self.tempresp_fun2(TPU_o, self.dHa_TPU, self.dHd_TPU, self.Topt_TPU, self.dHd_R_TPU)
        return TPU

    def getdS(self, tag: str):
        if self.type != 2:
            raise ValueError('No Topt fitted')

        # get the dHd based on tag
        if tag == 'Vcmax':
            dS_Vcmax = self.dHd_Vcmax/self.Topt_Vcmax + self.TRparam.R*torch.log(self.dHa_Vcmax/(self.dHd_Vcmax-self.dHa_Vcmax))
            return dS_Vcmax
        elif tag == 'Jmax':
            dS_Jmax = self.dHd_Jmax/self.Topt_Jmax + self.TRparam.R*torch.log(self.dHa_Jmax/(self.dHd_Jmax-self.dHa_Jmax))
            return dS_Jmax
        elif tag == 'TPU':
            dS_TPU = self.dHd_TPU/self.Topt_TPU + self.TRparam.R*torch.log(self.dHa_TPU/(self.dHd_TPU-self.dHa_TPU))
            return dS_TPU
        else:
            raise ValueError('tag should be Vcmax, Jmax or TPU')

    def setFitting(self, tag: str, fitting: bool):
        # get the self property based on tag
        try:
            param = getattr(self, tag)
        except AttributeError:
            raise ValueError('tag should be Vcmax, Jmax, TPU, Rd, Gamma, Kc or Ko')
        if isinstance(param, nn.Parameter):
            param.requires_grad = fitting



class FvCB(nn.Module):
    def __init__(self, lcd, LightResp_type :int = 0, TempResp_type : int = 1, onefit : bool = False, fitgm: bool = False, fitgamma: bool = False, fitKc: bool = False, fitKo: bool = False):
        super(FvCB, self).__init__()
        self.lcd = lcd
        self.Oxy = torch.tensor(213.5)
        self.LightResponse = LightResponse(self.lcd, LightResp_type)
        self.TempResponse = TemperatureResponse(self.lcd, TempResp_type)
        self.alphaG_r = nn.Parameter(torch.ones(self.lcd.num_FGs) * (-5))
        self.alphaG = None

        self.onefit = onefit
        if onefit:
            self.curvenum = self.lcd.num_FGs
        else:
            self.curvenum = self.lcd.num
        self.Vcmax25 = nn.Parameter(torch.ones(self.curvenum) * 100)
        self.Jmax25 = nn.Parameter(torch.ones(self.curvenum) * 200)
        self.TPU25 = nn.Parameter(torch.ones(self.curvenum) * 25)
        self.Rd25 = nn.Parameter(torch.ones(self.curvenum) * 1.5)

        self.Vcmax = None
        self.Jmax = None
        self.TPU = None
        self.Rd = None

        self.fitgm = fitgm
        if self.fitgm:
            self.gm = nn.Parameter(torch.ones(self.lcd.num_FGs))
        else:
            self.Cc = self.lcd.Ci

        self.fitgamma = fitgamma
        self.Gamma25 = torch.tensor(42.75).to(self.lcd.device)
        if self.fitgamma:
            self.Gamma25 = nn.Parameter(torch.ones(self.lcd.num_FGs).to(self.lcd.device) * self.Gamma25)
        else:
            self.Gamma = self.TempResponse.geGamma(self.Gamma25)
            
        if not self.fitgm and not self.fitgamma:
            self.Gamma_Cc = 1 - self.Gamma / self.Cc
            
        self.fitKc = fitKc
        self.Kc25 = torch.tensor(404.9).to(self.lcd.device)
        if self.fitKc:
            self.Kc25 = nn.Parameter(torch.ones(self.lcd.num_FGs).to(self.lcd.device) * self.Kc25)
        else:
            self.Kc = self.TempResponse.getKc(self.Kc25)
            
        self.fitKo = fitKo
        self.Ko25 = torch.tensor(278.4).to(self.lcd.device)
        if self.fitKo:
            self.Ko25 = nn.Parameter(torch.ones(self.lcd.num_FGs).to(self.lcd.device) * self.Ko25)
        else:
            self.Ko = self.TempResponse.getKo(self.Ko25)
            
        if not self.fitKc and not self.fitKo:
            self.Kco = self.Kc * (1 + self.Oxy / self.Ko)

    def expandparam(self, vcmax, jmax, tpu, rd):
        if self.onefit:
            if self.curvenum > 1:
                vcmax = torch.repeat_interleave(vcmax[self.lcd.FGs], self.lcd.lengths, dim=0)
                jmax = torch.repeat_interleave(jmax[self.lcd.FGs], self.lcd.lengths, dim=0)
                tpu = torch.repeat_interleave(tpu[self.lcd.FGs], self.lcd.lengths, dim=0)
                rd = torch.repeat_interleave(rd[self.lcd.FGs], self.lcd.lengths, dim=0)
        else:
            vcmax = torch.repeat_interleave(vcmax, self.lcd.lengths, dim=0)
            jmax = torch.repeat_interleave(jmax, self.lcd.lengths, dim=0)
            tpu = torch.repeat_interleave(tpu, self.lcd.lengths, dim=0)
            rd = torch.repeat_interleave(rd, self.lcd.lengths, dim=0)

        return vcmax, jmax, tpu, rd

    def forward(self):
        vcmax25, jmax25, tpu25, rd25 = self.expandparam(self.Vcmax25, self.Jmax25, self.TPU25, self.Rd25)

        self.Vcmax = self.TempResponse.getVcmax(vcmax25)
        self.Jmax = self.TempResponse.getJmax(jmax25)
        self.TPU = self.TempResponse.getTPU(tpu25)
        self.Rd = self.TempResponse.getRd(rd25)

        self.alphaG = torch.sigmoid(self.alphaG_r) * 3
        if self.lcd.num_FGs > 1:
            self.alphaG = torch.repeat_interleave(self.alphaG[self.lcd.FGs], self.lcd.lengths, dim=0)

        if self.fitgm:
            gm = torch.clamp(self.gm, min=0.0001)
            if self.lcd.num_FGs > 1:
                gm = torch.repeat_interleave(gm[self.lcd.FGs], self.lcd.lengths, dim=0)
            self.Cc = self.lcd.Ci - self.lcd.A / gm

        if self.fitgamma:
            if self.lcd.num_FGs > 1:
                gamma25 = torch.repeat_interleave(self.Gamma25[self.lcd.FGs], self.lcd.lengths, dim=0)
            else:
                gamma25 = self.Gamma25
            self.Gamma = self.TempResponse.geGamma(gamma25)

        if self.fitgm or self.fitgamma:
            self.Gamma_Cc = 1 - self.Gamma / self.Cc

        if self.fitKc:
            if self.lcd.num_FGs > 1:
                kc25 = torch.repeat_interleave(self.Kc25[self.lcd.FGs], self.lcd.lengths, dim=0)
            else:
                kc25 = self.Kc25
            self.Kc = self.TempResponse.getKc(kc25)

        if self.fitKo:
            if self.lcd.num_FGs > 1:
                ko25 = torch.repeat_interleave(self.Ko25[self.lcd.FGs], self.lcd.lengths, dim=0)
            else:
                ko25 = self.Ko25
            self.Ko = self.TempResponse.getKo(ko25)

        if self.fitKc or self.fitKo:
            self.Kco = self.Kc * (1 + self.Oxy / self.Ko)

        wc = self.Vcmax * self.Cc / (self.Cc + self.Kco)
        j = self.LightResponse.getJ(self.Jmax)
        wj = j * self.Cc / (4 * self.Cc + 8 * self.Gamma)
        cc_gamma = (self.Cc - self.Gamma * (1 + self.alphaG))
        cc_gamma = torch.clamp(cc_gamma, min=0.0001)
        wp = 3 * self.TPU * self.Cc / cc_gamma

        # w_min = torch.min(torch.stack((wc, wj, wp)), dim=0).values

        # a = self.Gamma_Cc * w_min - self.Rd
        ac = self.Gamma_Cc * wc - self.Rd
        aj = self.Gamma_Cc * wj - self.Rd
        ap = self.Gamma_Cc * wp - self.Rd
        a = torch.min(torch.stack((ac, aj, ap)), dim=0).values
        # gamma_all = (self.Gamma + self.Kco * self.Rd / self.Vcmax) / (1 - self.Rd / self.Vcmax)

        return a, ac, aj, ap



class correlationloss():
    def __init__(self, y):
        self.vy = y - torch.mean(y)
        self.sqvy = torch.sqrt(torch.sum(torch.pow(self.vy, 2)))
    def getvalue(self,x, targetR = 0.75):
        vx = x - torch.mean(x)
        cost = torch.sum(vx * self.vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * self.sqvy)

        if torch.isnan(cost):
            cost = torch.tensor(0.0)

        cost = torch.min(cost, torch.tensor(targetR))
        return (targetR - cost)

class Loss(nn.Module):
    def __init__(self, lcd, fitApCi: int = 500, fitCorrelation: bool = True):
        super().__init__()
        self.num_FGs = lcd.num_FGs
        self.mse = nn.MSELoss()
        self.end_indices = (lcd.indices + lcd.lengths - 1).long()
        self.A_r = lcd.A
        self.indices_end = (lcd.indices + lcd.lengths).long()
        self.indices_start = lcd.indices
        self.relu = nn.ReLU()
        self.mask_lightresp = lcd.mask_lightresp.bool()
        self.mask_nolightresp = ~self.mask_lightresp
        self.mask_fitAp = lcd.Ci[self.end_indices] > fitApCi # mask that last Ci is larger than specific value
        self.mask_fitAp = self.mask_fitAp.bool() & self.mask_nolightresp
        self.fitCorrelation = fitCorrelation

    def forward(self, fvc_model, An_o, Ac_o, Aj_o, Ap_o):

        # Reconstruction loss
        loss = self.mse(An_o, self.A_r) * 10

        if fvc_model.curvenum > 6 and self.fitCorrelation:
            corrloss = correlationloss(fvc_model.Vcmax25[self.mask_nolightresp])
            # make correlation between Jmax25 and Vcmax25 be 0.7
            loss += corrloss.getvalue(fvc_model.Jmax25[self.mask_nolightresp], targetR=0.7)
            # make correlation between Rd25 and 0.015*Vcmax25 be 0.4
            loss += corrloss.getvalue(fvc_model.Rd25[self.mask_nolightresp], targetR=0.4)
            # loss += self.mse(fvc_model.Rd25, 0.015 * fvc_model.Vcmax25) * 0.1

        if fvc_model.curvenum > 1:
            loss += torch.sum(self.relu(-fvc_model.Rd25))
        else:
            loss += self.relu(-fvc_model.Rd25)[0]

        if fvc_model.TempResponse.type != 0:
            if self.num_FGs > 1:
                loss += torch.sum(self.relu(-fvc_model.TempResponse.dHa_Vcmax)) * 10
                loss += torch.sum(self.relu(-fvc_model.TempResponse.dHa_Jmax))
                loss += torch.sum(self.relu(-fvc_model.TempResponse.dHa_TPU))
            elif self.num_FGs == 1:
                loss += self.relu(-fvc_model.TempResponse.dHa_Vcmax)[0] * 10
                loss += self.relu(-fvc_model.TempResponse.dHa_Jmax)[0]
                loss += self.relu(-fvc_model.TempResponse.dHa_TPU)[0]

        if fvc_model.TempResponse.type == 2:
            if self.num_FGs > 1:
                loss += torch.sum(self.relu(-fvc_model.TempResponse.Topt_Vcmax + fvc_model.TempResponse.TRparam.kelvin))
                loss += torch.sum(self.relu(-fvc_model.TempResponse.Topt_Jmax + fvc_model.TempResponse.TRparam.kelvin))
                loss += torch.sum(self.relu(-fvc_model.TempResponse.Topt_TPU + fvc_model.TempResponse.TRparam.kelvin))
            elif self.num_FGs == 1:
                loss += self.relu(-fvc_model.TempResponse.Topt_Vcmax + fvc_model.TempResponse.TRparam.kelvin)[0]
                loss += self.relu(-fvc_model.TempResponse.Topt_Jmax + fvc_model.TempResponse.TRparam.kelvin)[0]
                loss += self.relu(-fvc_model.TempResponse.Topt_TPU + fvc_model.TempResponse.TRparam.kelvin)[0]

        # penalty that Ap less than 0
        loss += torch.sum(self.relu(-Ap_o))

        # add constraint loss for last point
        # penalty that last Ap is larger than Ac and Aj
        penalty_pj = torch.clamp(Ap_o[self.end_indices] - Aj_o[self.end_indices], min=0)
        loss += torch.sum(penalty_pj[self.mask_fitAp]) * 0.15
        # penalty that last Aj is larger than Ac
        penalty_jc = torch.clamp(Aj_o[self.end_indices] - Ac_o[self.end_indices], min=0)
        loss += torch.sum(penalty_jc)

        Acj_o_diff = Ac_o - Aj_o
        Ajc_o_diff = -Acj_o_diff

        penalty_inter = torch.tensor(0.0)

        Acj_o_diff_abs = torch.abs(Acj_o_diff)
        Acj_o_diff = self.relu(Acj_o_diff)
        Ajc_o_diff = self.relu(Ajc_o_diff)

        for i in range(fvc_model.curvenum):

            index_start = self.indices_start[i]
            index_end = self.indices_end[i]

            # get the index that Ac closest to Aj
            index_closest = torch.argmin(Acj_o_diff_abs[index_start:index_end])
            Aj_inter = Aj_o[index_start+index_closest]
            Ap_inter = Ap_o[index_start+index_closest]

            # penalty that Ap is less than the intersection of Ac and Aj
            penalty_inter = penalty_inter + 5 * torch.clamp(Aj_inter * 1.1 - Ap_inter, min=0)

            # penalty to make sure part of Aj_o_i is larger than Ac_o_i
            ls_Aj_i = torch.sum(Ajc_o_diff[index_start:index_end])
            penalty_inter = penalty_inter + torch.clamp(8 - ls_Aj_i, min=0)

            ls_Ac_i = torch.sum(Acj_o_diff[index_start:index_end])
            penalty_inter = penalty_inter + torch.clamp(8 - ls_Ac_i, min=0)

        loss = loss + penalty_inter
        return loss
