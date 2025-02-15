import torch
import torch.nn as nn
import prospect.prospectmodels as prospect
def getAnthocyanin(refspectra, wavelength = torch.arange(400,2501,1)):
    rgreen = refspectra[:, (wavelength>540) & (wavelength<560)]
    rred = refspectra[:, (wavelength>690) & (wavelength<710)]
    rnir = refspectra[:, (wavelength>760) & (wavelength<800)]
    mari = (1/torch.mean(rgreen, 1) - 1 / torch.mean(rred, 1)) * torch.mean(rnir, 1)
    cant = 2.11 * mari + 0.45
    return cant.unsqueeze(1)

def run(prospectm: prospect.prospectdcore, refspectra32: torch.Tensor, transpectra32: torch.Tensor=None, learning_rate: float = 0.004, max_iter: int = 1000):
    device = torch.device('cpu')

    # set propc.cant gradient to False
    prospectm.cant.requires_grad = False
    criteria = prospect.Loss()

    params = list(prospectm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    loss_all = torch.tensor([], device=device)
    best_weights = prospectm.state_dict()
    best_iter = 0

    best_loss = float('inf')
    for iter in range(max_iter):

        optimizer.zero_grad()
        pred_specr, pred_trans = prospectm()
        if iter % 20 == 0:
            cantc = getAnthocyanin(pred_specr, torch.arange(400, 2501, 1))
            # set all negative values to 0
            cantc = torch.clamp(cantc, min=0)
            prospectm.cant = nn.Parameter(cantc, requires_grad=False)
        if transpectra32 is None:
            loss = criteria(prospectm, pred_specr, refspectra32)
        else:
            loss = criteria(prospectm, pred_specr, refspectra32, pred_trans, transpectra32)

        loss.backward()

        optimizer.step()
        # scheduler.step()

        loss_all = torch.cat((loss_all, loss.detach().unsqueeze(0)))

        if iter % 20 == 0:

            print(f'Iter {iter}, Loss: {loss.item()}')

        if loss < best_loss:
            best_loss = loss
            best_weights = prospectm.state_dict()
            best_iter = iter


        # if loss is nan, break
        if torch.isnan(loss):
            break
    print(f'Best loss: {best_loss} at iter {best_iter}')
    prospectm.load_state_dict(best_weights)
    return prospectm