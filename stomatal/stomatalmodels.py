import torch.nn as nn
import torch

class allparameters(nn.Module):
    def __init__(self):
        super(allparameters, self).__init__()

        self.Ca = torch.tensor(420.0)

class gsACi(nn.Module):
    def __init__(self,gs):
        super(gsACi, self).__init__()
        self.Ci = nn.Parameter(torch.ones(len(gs))*300)
        self.Ca = torch.tensor(420.0)
        self.gs = gs
    def forward(self):
        An = self.gs*(self.Ca - self.Ci)
        return An

class lossA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, An_fvcb, An_gs,Ci):
        loss = self.mse(An_fvcb, An_gs)
        loss += torch.sum(torch.relu(-Ci))*100
        return loss

# Ball Woodrow Berry
class BWB(nn.Module):
    def __init__(self, An, rh, lcpd = None):
        super(BWB, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.Ca = torch.tensor(420.0)
        self.A = An
        self.rh = rh
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.a1 = nn.Parameter(torch.ones(self.num_FGs))
    def forward(self):
        gs0 = self.gs0[self.FGs]
        a1 = self.a1[self.FGs]
        gs = gs0 + a1 * self.A * self.rh / self.Ca
        return gs

# Ball Berry Leuning
class BBL(nn.Module):
    def __init__(self, An, Gamma, VPD, lcpd = None):
        super(BBL, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.Gamma = Gamma
        self.VPD = VPD
        self.A = An
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.a1 = nn.Parameter(torch.ones(self.num_FGs))
        self.D0 = nn.Parameter(torch.ones(self.num_FGs))
        self.Ca = torch.tensor(420.0)
    def forward(self):
        gs0 = self.gs0[self.FGs]
        a1 = self.a1[self.FGs]
        D0 = self.D0[self.FGs]
        gs = gs0 + a1 * self.A / (self.Ca - self.Gamma) / (1 + self.VPD / D0)
        return gs

# Medlyn et al.
class MED(nn.Module):
    def __init__(self, An, VPD, lcpd = None):
        super(MED, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.A = An
        self.VPD = VPD
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.g1 = nn.Parameter(torch.ones(self.num_FGs))
        self.Ca = torch.tensor(420.0)
    def forward(self):
        gs0 = self.gs0[self.FGs]
        g1 = self.g1[self.FGs]
        gs = gs0 + 1.6 * (1 + g1 / torch.sqrt(self.VPD / 1000 * 101.3)) * self.A / self.Ca
        return gs

# Buckley Mott Farquhar
class BMF(nn.Module):
    def __init__(self, scd):
        super(BMF, self).__init__()
        self.model_label = 'BMF'
        self.num = scd.num
        self.Q = scd.Q
        self.VPD = scd.VPD
        self.Em = nn.Parameter(torch.ones(self.num))
        self.i0 = nn.Parameter(torch.ones(self.num)*10)
        self.k = nn.Parameter(torch.ones(self.num)*10000)
        self.b = nn.Parameter(torch.ones(self.num)*10)
        self.scd = scd
        self.lengths = scd.lengths
    def forward(self):
        self.Em_r = torch.repeat_interleave(self.Em, self.lengths)
        self.i0_r = torch.repeat_interleave(self.i0, self.lengths)
        self.k_r = torch.repeat_interleave(self.k, self.lengths)
        self.b_r = torch.repeat_interleave(self.b, self.lengths)
        self.Q_i0 = self.Q + self.i0_r
        self.Q_t_br = self.b_r  * self.Q
        self.Q_io_t_D = self.Q_i0 * self.VPD
        gs = self.Em_r * self.Q_i0 / (self.k_r + self.Q_t_br + self.Q_io_t_D)
        return gs
    def getpenalties(self):
        dgs_dQ = self.Em_r * ((self.k_r + self.b_r * self.Q + self.Q_io_t_D) - self.Q_i0 * (self.b_r  + self.VPD)) / (self.k_r + self.Q_t_br + self.Q_io_t_D) ** 2
        dgs_dD = self.Em_r * (-self.Q_i0 ** 2) / (self.k_r + self.Q_t_br + self.Q_io_t_D) ** 2
        d2gs_dQ2 = - (2 * self.Em_r * (self.k_r - self.i0_r * self.b_r) * (self.b_r + self.VPD)) / (self.k_r + self.Q_t_br + self.Q_io_t_D) ** 3
        return dgs_dQ, dgs_dD, d2gs_dQ2

class lossSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, scm,gs_fit):
        loss = self.mse(gs_fit, scm.scd.gsw) * 10
        # get all learnable parameters in scm
        for param in scm.parameters():
            loss += torch.sum(torch.relu(-param))*10
        if scm.model_label == 'BMF':
            dgs_dQ, dgs_dD, d2gs_dQ2 = scm.getpenalties()
            loss += torch.sum(torch.relu(-dgs_dQ))
            loss += torch.sum(torch.relu(dgs_dD))

        return loss
