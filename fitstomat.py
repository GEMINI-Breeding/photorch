import torch
import stomatalmodels as stomat

def getACi(fvcbmtt, gsw, learnrate = 2, maxiteration = 8000, minloss = 1e-10):
    gsmtest = stomat.gsACi(torch.tensor(gsw))
    optimizer = torch.optim.Adam(gsmtest.parameters(), lr=learnrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 2000, gamma=0.9)
    best_loss = 100000
    best_iter = 0
    best_weights = gsmtest.state_dict()
    criterion = stomat.lossA()
    minloss = minloss
    for iter in range(maxiteration):

        optimizer.zero_grad()
        An_gs = gsmtest()
        fvcbmtt.lcd.Ci = gsmtest.Ci
        fvcbmtt.lcd.A = An_gs
        An_f, Ac_o, Aj_o, Ap_o = fvcbmtt()
        loss = criterion(An_f, An_gs, gsmtest.Ci)

        loss.backward()
        if (iter + 1) % 100 == 0:
            # print(vcmax25)
            print(f'Loss at iter {iter}: {loss.item():.4f}')

        optimizer.step()
        scheduler.step()

        if loss.item() < minloss:
            best_loss = loss.item()
            best_weights = gsmtest.state_dict()
            best_iter = iter
            print(f'Fitting converged at iter {iter}: {loss.item():.4f}')
            break

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = gsmtest.state_dict()
            best_iter = iter
    print(f'Best loss at iter {best_iter}: {best_loss:.4f}')
    gsmtest.load_state_dict(best_weights)
    return gsmtest


def run(scm, gsw, learnrate = 0.01, maxiteration = 8000, minloss = 1e-6):
    optimizer = torch.optim.Adam(scm.parameters(), lr=learnrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    best_loss = 100000
    best_iter = 0
    best_weights = scm.state_dict()
    criterion = stomat.lossSC()

    for iter in range(maxiteration):

        optimizer.zero_grad()
        gs_fit = scm()
        loss = criterion(scm,gs_fit,gsw)

        loss.backward()
        if (iter + 1) % 100 == 0:
            # print(vcmax25)
            print(f'Loss at iter {iter}: {loss.item():.4f}')

        optimizer.step()
        scheduler.step()

        if loss.item() < minloss:
            best_loss = loss.item()
            best_weights = scm.state_dict()
            best_iter = iter
            print(f'Fitting converged at iter {iter}: {loss.item():.4f}')
            break

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = scm.state_dict()
            best_iter = iter
    print(f'Best loss at iter {best_iter}: {best_loss:.4f}')
    scm.load_state_dict(best_weights)
    return scm