# PhoTorch
# A/Ci curve optimizer
import torch
import fvcbmodels as initM
import initphotodata as initD
import time

# get rmse loss
def get_rmse_loss(An_o, An_r):
    rmse = torch.sqrt(torch.mean((An_o - An_r) ** 2))
    return rmse

class modelresult():
    def __init__(self, fvcbm_fit: initM.FvCB, loss_all: torch.tensor, allweights: dict = None):
        self.model = fvcbm_fit
        self.losses = loss_all
        self.recordweights = allweights

def run(fvcbm:initM.FvCB, learn_rate = 0.6, device= 'cpu', maxiteration = 8000, minloss = 3, recordweightsTF = False, fitcorr = False, ApCithreshold = 500):
    start_time = time.time()

    if device == 'cuda':
        device = torch.device(device)
        fvcbm.to(device)
        loss_all = torch.tensor([]).to(device)
    else:
        loss_all = torch.tensor([])

    criterion = initM.Loss(fvcbm.lcd, ApCithreshold, fitcorr)
    optimizer = torch.optim.Adam(fvcbm.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5000, gamma=0.8)

    best_loss = 100000

    best_weights = fvcbm.state_dict()
    best_iter = 0

    class recordweights:
        def __init__(self):
            self.allweights = {}
        def getweights(self, model):
            for name, param in model.named_parameters():
                if name not in self.allweights:
                    self.allweights[name] = param.data.cpu().unsqueeze(0)
                else:
                    self.allweights[name] = torch.cat((self.allweights[name], param.data.cpu().unsqueeze(0)), dim=0)
            # add alphaG to the record
            self.allweights['alphaG'] = model.alphaG.data.cpu().unsqueeze(0)

    recordweights = recordweights()

    for iter in range(maxiteration):

        optimizer.zero_grad()

        An_o, Ac_o, Aj_o, Ap_o = fvcbm()
        loss = criterion(fvcbm, An_o, Ac_o, Aj_o, Ap_o)

        loss.backward()
        if (iter + 1) % 200 == 0:
            print(f'Loss at iter {iter}: {loss.item():.4f}')

        optimizer.step()
        scheduler.step()
        if recordweightsTF:
            recordweights.getweights(fvcbm)
        loss_all = torch.cat((loss_all, loss.unsqueeze(0)), dim=0)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = fvcbm.state_dict()
            best_iter = iter

        if loss.item() < minloss:
            print(f'Fitting stopped at iter {iter}')
            break

    print(f'Best loss at iter {best_iter}: {best_loss:.4f}')

    fvcbm.load_state_dict(best_weights)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Fitting time: {elapsed_time:.4f} seconds')
    if recordweightsTF:
        modelresult_out = modelresult(fvcbm, loss_all, recordweights.allweights)
    else:
        modelresult_out = modelresult(fvcbm, loss_all, None)

    return modelresult_out

def getVadlidTPU(fvcbm:initM.FvCB, threshold_jp: float = 0.5):

    A, Ac, Aj, Ap = fvcbm()
    IDs = fvcbm.lcd.IDs

    last2diff = Aj[fvcbm.lcd.indices + fvcbm.lcd.lengths-1]-Ap[fvcbm.lcd.indices + fvcbm.lcd.lengths-1]
    mask_vali = last2diff > threshold_jp
    mask_invali = last2diff < threshold_jp

    for i in range(len(IDs)):
        indices = fvcbm.lcd.getIndicesbyID(IDs[i])
        if mask_invali[i]:
            Ap[indices] = Ap[indices] + 1000

    A_new =  torch.min(torch.stack((Ac, Aj, Ap)), dim=0).values
    return A_new, mask_vali


