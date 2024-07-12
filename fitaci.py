import torch
import initmodels as initM
import initdata as initD
import time

# get rmse loss
def get_rmse_loss(An_o, An_r):
    rmse = torch.sqrt(torch.mean((An_o - An_r) ** 2))
    return rmse

def run(fvcbm:initM.FvCB, learn_rate = 0.6, device= 'cpu', maxiteration = 8000, minloss = 3, recordweightsTF = False):
    start_time = time.time()

    if device == 'cuda':
        device = torch.device(device)
        fvcbm.to(device)
        loss_all = torch.tensor([]).to(device)
    else:
        loss_all = torch.tensor([])

    criterion = initM.Loss(fvcbm.lcd)
    optimizer = torch.optim.Adam(fvcbm.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)

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

    recordweights = recordweights()

    for iter in range(maxiteration):

        optimizer.zero_grad()

        An_o, Ac_o, Aj_o, Ap_o, _ = fvcbm()
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
        return fvcbm, recordweights.allweights
    else:
        return fvcbm

def getVadlidTPU(fvcbm:initM.FvCB, lcd:initD.initLicordata, threshold_jp: float = 0.5):

    A, Ac, Aj, Ap, _ = fvcbm()
    IDs = lcd.IDs

    last2diff = Aj[lcd.indices+lcd.lengths-1]-Ap[lcd.indices+lcd.lengths-1]
    mask_vali = last2diff > threshold_jp
    mask_invali = last2diff < threshold_jp

    for i in range(len(IDs)):
        indices = lcd.getIndicesbyID(IDs[i])
        if mask_invali[i]:
            Ap[indices] = Ap[indices] + 1000

    A_new =  torch.min(torch.stack((Ac, Aj, Ap)), dim=0).values
    return A_new, mask_vali
