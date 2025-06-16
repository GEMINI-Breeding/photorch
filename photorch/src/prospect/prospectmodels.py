import numpy as np
import torch
import torch.nn as nn
from scipy.special import expi
from torch.autograd import Function

class ExpiFunction(Function):
    @staticmethod
    def forward(ctx, x):
        x_numpy = x.detach().cpu().numpy()  # Convert tensor to numpy
        result = torch.tensor(expi(x_numpy), dtype=x.dtype, device=x.device)  # Apply expi and convert back to tensor
        ctx.save_for_backward(x)  # Save input for backward computation
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors  # Retrieve saved tensor
        grad_input = grad_output * torch.exp(x) / x  # Compute gradient
        return grad_input

class expiTorch(nn.Module):
    def __init__(self):
        super(expiTorch, self).__init__()

    def forward(self, x):
        return ExpiFunction.apply(x)


class getav(nn.Module):
    def __init__(self):
        super(getav, self).__init__()


    def forward(self, alpha, nr):
        n2 = torch.pow(nr, 2)
        npx = n2 + 1
        nm = n2 - 1
        a = (nr + 1) * (nr + 1) / 2.
        k = -(n2 - 1) * (n2 - 1) / 4.
        rd = alpha * (torch.pi / 180)
        sa = torch.sin(rd)
        sa2 = torch.pow(sa, 2)
        if alpha != 90:
            b1 = torch.sqrt(torch.pow(sa2 - npx / 2, 2) + k)
        else:
            b1 = 0.

        b2 = sa2 - npx / 2
        b = b1 - b2
        k2 = torch.pow(k, 2)
        ts = (k2 / (6 * torch.pow(b, 3)) + k / b - b / 2) - (k2 / (6 * torch.pow(a, 3)) + k / a - a / 2)

        tp1 = -2 * n2 * (b - a) / torch.pow(npx, 2)

        nm2 = torch.pow(nm, 2)
        tp2 = -2 * n2 * npx * torch.log(b / a) / nm2

        tp3 = n2 * (1 / b - 1 / a) / 2

        n22 = torch.pow(n2, 2)
        npx3 = torch.pow(npx, 3)
        npax_a = 2 * npx * a - nm2
        npx_b = 2 * npx * b - nm2
        tp4 = 16 * n22 * (n22 + 1) * torch.log(npx_b / npax_a) / (npx3 * nm2)

        tp5 = 16 * torch.pow(n2, 3) * (1. / npx_b - 1 / npax_a) / npx3

        tp = tp1 + tp2 + tp3 + tp4 + tp5
        tav = (ts + tp) / (2 * sa2)

        return tav

class reftrans_onelayer(nn.Module):
    def __init__(self):
        super(reftrans_onelayer, self).__init__()
        self.getav = getav()

    def forward(self, alpha, nr, tau):
        talf = self.getav(alpha, nr)
        ralf = 1.0 - talf
        t12 = self.getav(torch.tensor([90.]), nr)

        r12 = 1. - t12
        t21 = t12 / (nr * nr)
        r21 = 1 - t21

        # top surface side
        denom = 1. - r21 * r21 * tau * tau
        Ta = talf * tau * t21 / denom
        Ra = ralf + r21 * tau * Ta

        # bottom surface side
        t = t12 * tau * t21 / denom
        r = r12 + r21 * tau * t

        return r, t, Ra, Ta, denom

class prospectdcore(nn.Module):
    def __init__(self, num_leaves=1):
        super(prospectdcore, self).__init__()
        self.N = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 1.2)
        self.cab = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 30)  # chlorophyll content (mu g/cm2)
        self.car = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 10)  # carotenoid content (mu g/cm2)
        # self.cbrown = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 0.00001)  # brown pigment content (arbitrary units)
        self.water = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 0.005)  # equivalent water thickness (cm)
        self.lma = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 0.02)  # leaf mass per unit area (g/cm2)
        self.cant = nn.Parameter(torch.ones(num_leaves, 1, dtype=torch.float) * 1.0)  # anthocyanin content (mu g/cm2)

        _, nr_d, kab_d, kcar_d, kant_d, kbrown_d, kw_d, km_d = np.loadtxt('prospect/prospectd_absc.txt', unpack=True)
        self.nr = torch.tensor(nr_d, dtype=torch.float)
        self.kab = torch.tensor(kab_d, dtype=torch.float)
        self.kcar = torch.tensor(kcar_d, dtype=torch.float)
        self.kant = torch.tensor(kant_d, dtype=torch.float)
        # self.kbrown = torch.tensor(kbrown_d, dtype=torch.float)
        self.kw = torch.tensor(kw_d, dtype=torch.float)
        self.km = torch.tensor(km_d, dtype=torch.float)
        self.lambdas = torch.arange(400, 2501).float()
        self.n_lambda = len(self.lambdas)
        self.getav = getav()
        self.alpha = torch.tensor([40.], dtype=torch.float)
        self.expi = expiTorch()
        self.getonelayer = reftrans_onelayer()

    def forward(self):
        # kall = (self.cab * self.kab + self.car * self.kcar + self.cant * self.kant + self.cbrown * self.kbrown + self.water * self.kw + self.lma * self.km) / self.N
        kall = (self.cab * self.kab + self.car * self.kcar + self.cant * self.kant + self.water * self.kw + self.lma * self.km) / self.N
        kall = torch.clamp(kall, min=0.0001)
        t1 = (1 - kall) * torch.exp(-kall)
        t2 = torch.pow(kall, 2) * (-self.expi(-kall))

        tau = t1 + t2

        r, t, Ra, Ta, denom = self.getonelayer(self.alpha, self.nr, tau)
        rpt = r + t
        rmt = r - t
        D = torch.sqrt((1 + rpt) * (1 + rmt) * (1. - rmt) * (1. - rpt))
        rq = torch.pow(r, 2)
        tq = torch.pow(t, 2)
        rqmtq = rq - tq
        a = (1 + rqmtq + D) / (2 * r)
        b = (1 - rqmtq + D) / (2 * t)
        bnm1 = torch.pow(b, self.N - 1)
        bn2 = torch.pow(bnm1, 2)
        a2 = a * a
        denom = a2 * bn2 - 1
        Rsub = a * (bn2 - 1) / denom
        Tsub = bnm1 * (a2 - 1) / denom

        # Case of zero absorption
        j = r + t >= 1.
        if torch.sum(j) > 0:
            N_repeat = self.N.expand(-1, t.shape[1])
            Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N_repeat[j] - 1))
            Rsub[j] = 1 - Tsub[j]

        # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
        denom = 1 - Rsub * r
        tran = Ta * Tsub / denom
        refl = Ra + Ta * Rsub * t / denom

        return refl, tran

def correlationloss(x, y, target_r = 0.7):

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        sqvy = torch.sqrt(torch.sum(torch.pow(vy, 2)))
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * sqvy)

        if torch.isnan(cost):
            cost = torch.tensor(0.0)

        cost = torch.min(cost, torch.tensor(target_r))
        return (target_r - cost)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, promodel, pred_ref, target_ref, pred_tran=None, target_tran=None):

        # add penalty if N smaller than 1
        loss = torch.sum(torch.clamp(1-promodel.N, min=0))
        # add penalty if N larger than 2.5
        loss += torch.sum(torch.clamp(promodel.N-2.5, min=0))
        # add penalty if cab less than 0
        loss += torch.sum(torch.clamp(-promodel.cab, min=0))

        # add penalty if car less than 0
        loss += torch.sum(torch.clamp(-promodel.car, min=0))
        # add penalty if cant less than 0
        loss += torch.sum(torch.clamp(-promodel.cant, min=0))

        # add penalty if cbrown less than 0
        # loss += torch.sum(torch.clamp(-promodel.cbrown, min=0))

        # add penalty if water less than 0
        loss += torch.sum(torch.clamp(0.004-promodel.water, min=0))
        loss += torch.sum(torch.clamp(promodel.water-0.04, min=0))

        # add penalty if LMA less than 0.0015
        loss += torch.sum(torch.clamp(0.0015-promodel.lma, min=0))
        # add penalty if LMA larger than 0.04
        loss += torch.sum(torch.clamp(promodel.lma-0.04, min=0))

        # add penalty if correlation between car and cab is less than 0.9
        if len(promodel.car) > 8:
            loss += correlationloss(promodel.car,promodel.cab, 0.9)
            loss += correlationloss(promodel.water, promodel.lma, 0.6)

        loss += self.mse(pred_ref[:, :-150], target_ref[:, :-100])
        if pred_tran is not None and target_tran is not None:
            loss += self.mse(pred_tran[:, :-150], target_tran[:, :-100])

        # if pred_tran is not None and target_tran is None:
        #     correlationrt = correlationloss1(pred_ref[:, :-150])
        #     loss += correlationrt.getvalue(pred_tran[:, :-150],0.9)

        return loss
    

class fitparams(nn.Module):
    def __init__(self, num_leaves=1):
        super().__init__()

        _, nr_d, kab_d, kcar_d, kant_d, kbrown_d, kw_d, km_d = np.loadtxt('prospect/prospectd_absc.txt', unpack=True)
        self.nr = nn.Parameter(torch.tensor(nr_d, dtype=torch.float))
        self.kab = nn.Parameter(torch.tensor(kab_d, dtype=torch.float))
        self.kcar = nn.Parameter(torch.tensor(kcar_d, dtype=torch.float))
        self.kant = nn.Parameter(torch.tensor(kant_d, dtype=torch.float))
        # self.kbrown = nn.Parameter(torch.tensor(kbrown_d, dtype=torch.float))
        self.kw = nn.Parameter(torch.tensor(kw_d, dtype=torch.float))
        self.km = nn.Parameter(torch.tensor(km_d, dtype=torch.float))
        self.lambdas = torch.arange(400, 2501).float()
        self.n_lambda = len(self.lambdas)
        self.getav = getav()
        self.alpha = torch.tensor([40.], dtype=torch.float)
        self.expi = expiTorch()
        self.getonelayer = reftrans_onelayer()

    def forward(self,N,cab,car,water,lma,cant):
        kall = (cab * self.kab + car * self.kcar + cant * self.kant + water * self.kw + lma * self.km) / N
        kall = torch.clamp(kall, min=0.0001)
        t1 = (1 - kall) * torch.exp(-kall)
        t2 = torch.pow(kall, 2) * (-self.expi(-kall))

        tau = t1 + t2

        r, t, Ra, Ta, denom = self.getonelayer(self.alpha, self.nr, tau)
        rpt = r + t
        rmt = r - t
        D = torch.sqrt((1 + rpt) * (1 + rmt) * (1. - rmt) * (1. - rpt))
        rq = torch.pow(r, 2)
        tq = torch.pow(t, 2)
        rqmtq = rq - tq
        a = (1 + rqmtq + D) / (2 * r)
        b = (1 - rqmtq + D) / (2 * t)
        bnm1 = torch.pow(b, self.N - 1)
        bn2 = torch.pow(bnm1, 2)
        a2 = a * a
        denom = a2 * bn2 - 1
        Rsub = a * (bn2 - 1) / denom
        Tsub = bnm1 * (a2 - 1) / denom

        # Case of zero absorption
        j = r + t >= 1.
        if torch.sum(j) > 0:
            N_repeat = self.N.expand(-1, t.shape[1])
            Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N_repeat[j] - 1))
            Rsub[j] = 1 - Tsub[j]

        # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
        denom = 1 - Rsub * r
        tran = Ta * Tsub / denom
        refl = Ra + Ta * Rsub * t / denom

        return refl, tran
    

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, promodel, pred_ref, target_ref, pred_tran=None, target_tran=None):


        loss += self.mse(pred_ref[:, :-150], target_ref[:, :-100])
        if pred_tran is not None and target_tran is not None:
            loss += self.mse(pred_tran[:, :-150], target_tran[:, :-100])


        return loss
