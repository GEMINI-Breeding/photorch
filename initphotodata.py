# PhoTorch
# Licor data initialization
import torch
import numpy as np

def remove_ud_trend(A, Ci, up_treshold = 0.06, down_treshold = 0.06, keepCi = 1000):
    # Calculate the difference between consecutive data points
    diff = np.diff(A)
    indices = np.arange(len(A))
    # Find the point where the upward trend starts
    upward_trend_start = None
    for i in range(len(diff) - 1, -1, -1):
        if Ci[i] < keepCi:
            break
        if diff[i] > up_treshold:
            upward_trend_start = i
            continue
        else:
            break

    # If an upward trend is found, remove the upward segment
    if upward_trend_start is not None:
        indices = indices[:upward_trend_start + 1]

    downward_trend_start = None
    for i in range(len(diff) - 1, -1, -1):
        if Ci[i] < keepCi:
            break
        if diff[i] < -down_treshold:
            downward_trend_start = i
            continue
        else:
            break
    if downward_trend_start is not None:
        indices = indices[:downward_trend_start + 1]
    return indices

def preprocessCurve(A, Ci, indices, smoothingwindow = 10, up_treshold=0.06, down_treshold=0.06, lightresp = False):

    from scipy.signal import savgol_filter

    if not lightresp:
        if len(A[Ci > 600]) > smoothingwindow*3:
            A[Ci > 600] = savgol_filter(A[Ci > 600], smoothingwindow, 1)

        indices_nup = remove_ud_trend(A, Ci, up_treshold, down_treshold, 1000)
        A = A[indices_nup]
        Ci = Ci[indices_nup]
        indices = indices[indices_nup]

    # cut off the data points where Ci < 0
    # A = A[Ci > 0]
    # indices = indices[Ci > 0]
    # Ci = Ci[Ci > 0]
    # cut off the data points where Ci > 3000
    A = A[Ci < 3000]
    indices = indices[Ci < 3000]
    Ci = Ci[Ci < 3000]

    minCi_index = np.argmin(Ci)
    A_cimin = A[minCi_index]
    minA_index = np.argmin(A)
    Ci_amin = Ci[minA_index]

    if not lightresp:
        # drop data points where Ci < Ci_amin and A < A_cimin
        indices_lci = (Ci > Ci_amin) | (A > A_cimin)
        # if only one data point is left, keep it
        if len(indices) - np.sum(indices_lci) > 1:
            indices = indices[indices_lci]
            A = A[indices_lci]
            Ci = Ci[indices_lci]

    return A, Ci, indices

class initLicordata():
    def __init__(self, LCdata, preprocess = True, lightresp_id = None, smoothingwindow = 10, up_treshold=0.06, down_treshold=0.06):
        idname = 'CurveID'
        all_IDs = LCdata[idname].values
        self.device = 'cpu'
        IDs = np.unique(all_IDs)

        fgname = 'FittingGroup'
        try:
            all_FGs = LCdata[fgname].values
            FGs_uq = np.unique(all_FGs)
        except:
            raise ValueError('FittingGroup not found in the data')

        self.IDs = np.array([])
        self.FGs_idx = np.array([])
        self.FGs_name = np.array([])

        self.A = torch.empty((0,))  # net photosynthesis
        self.Q = torch.empty((0,)) # PPFD
        self.Ci = torch.empty((0,)) # intercellular CO2
        self.Tleaf = torch.empty((0,)) # leaf temperature

        # self.gsw = torch.empty((0,)) # stomatal conductance
        # self.Ca = torch.empty((0,)) # ambient CO2
        # self.rh = torch.empty((0,)) # air relative humidity
        # self.D = torch.empty((0,)) # vapor pressure deficit

        idx = torch.tensor([0])
        sample_indices = torch.empty((0,), dtype=torch.int32)
        sample_lengths = torch.empty((0,), dtype=torch.int32)

        # create a boolean mask for curve fitting, initialize all to True with length equal to the number of samples
        self.mask_lightresp = torch.tensor([])

        for i in range(len(IDs)):
            id = IDs[i]
            indices = np.where(LCdata[idname] == id)[0]

            # smooth A values where Ci > 500
            A = LCdata['A'].iloc[indices].to_numpy()
            Ci = LCdata['Ci'].iloc[indices].to_numpy()

            sorted_indices = np.argsort(Ci)
            A = A[sorted_indices]
            Ci = Ci[sorted_indices]
            indices = indices[sorted_indices]

            # if there are Ci less than 0
            if np.sum(Ci < 0) > 0:
                print('Warning: Found Ci < 0 in ID:', id, ', removing this A/Ci curve')
                continue

            self.IDs = np.append(self.IDs, id)
            fg = LCdata[fgname].iloc[indices[0]]
            self.FGs_name = np.append(self.FGs_name, fg)
            # # get the idex of the fg in FGs_uq
            # fg_idx = np.where(FGs_uq == fg)[0][0]
            # self.FGs_idx = np.append(self.FGs_idx, fg_idx)

            if lightresp_id is not None and id in lightresp_id:
                self.mask_lightresp = torch.cat((self.mask_lightresp, torch.tensor([True])))
                lightcurve = True
            else:
                self.mask_lightresp = torch.cat((self.mask_lightresp, torch.tensor([False])))
                lightcurve = False
            if preprocess:
                A, Ci, indices = preprocessCurve(A, Ci, indices, smoothingwindow, up_treshold, down_treshold, lightcurve)

            self.A = torch.cat((self.A, torch.tensor(A)))
            try:
                self.Q = torch.cat((self.Q, torch.tensor(LCdata['Qin'].iloc[indices].to_numpy())))
            except:
                # fill Q with default value 2000
                self.Q = torch.cat((self.Q, torch.tensor([2000]*len(indices))))
                print('Warning: Qin not found, filling with default value 2000')
            self.Ci = torch.cat((self.Ci, torch.tensor(Ci)))
            try:
                self.Tleaf = torch.cat((self.Tleaf,torch.tensor(LCdata['Tleaf'].iloc[indices].to_numpy() + 273.15)))
            except:
                # fill Tleaf with default value 25
                self.Tleaf = torch.cat((self.Tleaf, torch.tensor([25+273.15]*len(indices))))
                print('Warning: Tleaf not found, filling with default value 25 C (298.15 K)')

            # self.gsw = torch.cat((self.gsw, torch.tensor(LCdata['gsw'].iloc[indices].to_numpy())))
            # self.Ca = torch.cat((self.Ca, torch.tensor(LCdata['Ca'].iloc[indices].to_numpy())))
            # self.rh = torch.cat((self.rh, torch.tensor(LCdata['RHcham'].iloc[indices].to_numpy() / 100)))
            # self.D = torch.cat((self.D, torch.tensor(LCdata['VPDleaf'].iloc[indices].to_numpy() / LCdata['Pa'].iloc[indices].to_numpy() * 1000)))

            sample_indices = torch.cat((sample_indices, idx))
            idx += len(indices)
            sample_lengths = torch.cat((sample_lengths, torch.tensor([len(indices)], dtype=torch.int32)))


        FGs_uq = np.unique(self.FGs_name)
         # get the idex of the FGs_name in FGs_uq
        for fg in self.FGs_name:
            fg_idx = np.where(FGs_uq == fg)[0][0]
            self.FGs_idx = np.append(self.FGs_idx, fg_idx)

        self.num_FGs = len(FGs_uq)
        self.indices = sample_indices
        self.lengths = sample_lengths
        self.num = len(self.IDs)

        # print done reading data information
        print('Done reading:', self.num, 'A/Ci curves;', len(self.A), 'data points')

    def todevice(self, device: torch.device = 'cpu'):
        self.device = device
        self.A = self.A.to(device)
        self.Q = self.Q.to(device)
        self.Ci = self.Ci.to(device)
        self.Tleaf = self.Tleaf.to(device)
        # self.gsw = self.gsw.to(device)
        # self.Ca = self.Ca.to(device)
        # self.rh = self.rh.to(device)
        # self.D = self.D.to(device)
        self.indices = self.indices.to(device)
        self.lengths = self.lengths.to(device)
        self.mask_lightresp = self.mask_lightresp.to(device)

    def getDatabyID(self, ID):
        # get the index of ID
        idx_ID = np.where(self.IDs == ID)[0][0]
        index_start = self.indices[idx_ID].int()
        index_end = (self.indices[idx_ID] + self.lengths[idx_ID]).int()
        A = self.A[index_start:index_end].cpu().numpy()
        Ci = self.Ci[index_start:index_end].cpu().numpy()
        Q = self.Q[index_start:index_end].cpu().numpy()
        Tleaf = self.Tleaf[index_start:index_end].cpu().numpy()
        return A, Ci, Q, Tleaf

    def getIndicesbyID(self, ID):
        try:
            idx_ID = np.where(self.IDs == ID)[0][0]
        except:
            raise ValueError('ID', ID, 'not found')
        index_start = self.indices[idx_ID].int()
        index_end = (self.indices[idx_ID] + self.lengths[idx_ID]).int()
        indices = np.arange(index_start.cpu(), index_end.cpu())
        return indices

    def getFitGroupbyID(self, ID):
        try:
            fg_idx = self.FGs_idx[np.where(self.IDs == ID)[0][0]]
            fg_name = self.FGs_name[np.where(self.IDs == ID)[0][0]]
        except:
            raise ValueError('ID', ID, 'not found')
        return fg_name, fg_idx
