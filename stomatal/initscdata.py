import numpy as np
import torch

class initscdata():
    def __init__(self, LCdata, printout = True):
        idname = 'CurveID'
        all_IDs = LCdata[idname].values
        self.device = 'cpu'
        IDs = np.unique(all_IDs)

        fgname = 'FittingGroup'
        try:
            all_FGs = LCdata[fgname].values
            FGs_uq = np.unique(all_FGs)
        except:
            # add a new column 'FittingGroup' with all values equal to 1
            LCdata['FittingGroup'] = 1
            # if printout:
            #     print('Warning: FittingGroup not found in the data, adding a FittingGroup column with all values equal to 1')

        self.IDs = np.array([])
        self.FGs_idx = np.array([])
        self.FGs_name = np.array([])
        self.hasA = True
        # check if LCdata has 'A' column
        if 'A' not in LCdata.columns:
            self.hasA = False
            if printout:
                print('Warning: A column not found in the data')
        else:
            self.A = torch.empty((0,))  # net photosynthesis
        self.Q = torch.empty((0,)) # PPFD
        # self.Ci = torch.empty((0,)) # intercellular CO2
        self.Tleaf = torch.empty((0,)) # leaf temperature

        self.gsw = torch.empty((0,)) # stomatal conductance
        # self.Ca = torch.empty((0,)) # ambient CO2
        self.rh = torch.empty((0,)) # air relative humidity
        self.VPD = torch.empty((0,)) # vapor pressure deficit

        idx = torch.tensor([0])
        sample_indices = torch.empty((0,), dtype=torch.int32)
        sample_lengths = torch.empty((0,), dtype=torch.int32)

        for i in range(len(IDs)):
            id = IDs[i]
            indices = np.where(LCdata[idname] == id)[0]

            if self.hasA:
                A = LCdata['A'].iloc[indices].to_numpy()
                self.A = torch.cat((self.A, torch.tensor(A)))
            # Ci = LCdata['Ci'].iloc[indices].to_numpy()
            # self.Ci = torch.cat((self.Ci, torch.tensor(Ci)))

            # sorted_indices = np.argsort(Ci)
            # A = A[sorted_indices]
            # Ci = Ci[sorted_indices]
            # indices = indices[sorted_indices]

            # if there are Ci less than 0
            # if np.sum(Ci < 0) > 0 and printout:
            #     print('Warning: Found Ci < 0 in ID:', id, ', removing this data')
            #     continue

            self.IDs = np.append(self.IDs, id)
            fg = LCdata[fgname].iloc[indices[0]]
            self.FGs_name = np.append(self.FGs_name, fg)
            # # get the idex of the fg in FGs_uq
            # fg_idx = np.where(FGs_uq == fg)[0][0]
            # self.FGs_idx = np.append(self.FGs_idx, fg_idx)
            leaf_PAR_absorptivity = 0.85

            self.Tleaf = torch.cat((self.Tleaf,torch.tensor(LCdata['Tleaf'].iloc[indices].to_numpy() + 273.15)))

            self.gsw = torch.cat((self.gsw, torch.tensor(LCdata['gsw'].iloc[indices].to_numpy())))
            # self.Ca = torch.cat((self.Ca, torch.tensor(LCdata['Ca'].iloc[indices].to_numpy())))

            if 'RHcham' in LCdata.columns:
                self.rh = torch.cat((self.rh, torch.tensor(LCdata['RHcham'].iloc[indices].to_numpy() / 100)))
                self.Q = torch.cat((self.Q, torch.tensor(LCdata['Qin'].iloc[indices].to_numpy()*leaf_PAR_absorptivity)))

            else:
                raise Exception("No valid relative humidity column header found. Accepted header should be 'RHcham'")
            # self.D = torch.cat((self.D, torch.tensor(LCdata['VPDleaf'].iloc[indices].to_numpy() / LCdata['Pa'].iloc[indices].to_numpy() * 1000)))
            self.VPD = torch.cat((self.VPD, torch.tensor(LCdata['VPDleaf'].iloc[indices].to_numpy())))


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

        if printout:
            # print done reading data information
            print('Done reading:', self.num, 'gsw curves;', len(self.VPD), 'data points')

    def todevice(self, device: torch.device = 'cpu'):
        self.device = device
        self.A = self.A.to(device)
        self.Q = self.Q.to(device)
        # self.Ci = self.Ci.to(device)
        self.Tleaf = self.Tleaf.to(device)
        self.gsw = self.gsw.to(device)
        # self.Ca = self.Ca.to(device)
        self.rh = self.rh.to(device)
        self.VPD = self.VPD.to(device)
        self.indices = self.indices.to(device)
        self.lengths = self.lengths.to(device)

    def getDatabyID(self, ID):
        # get the index of ID
        idx_ID = np.where(self.IDs == ID)[0][0]
        index_start = self.indices[idx_ID].int()
        index_end = (self.indices[idx_ID] + self.lengths[idx_ID]).int()
        A = self.A[index_start:index_end].cpu().numpy()
        # Ci = self.Ci[index_start:index_end].cpu().numpy()
        Q = self.Q[index_start:index_end].cpu().numpy()
        Tleaf = self.Tleaf[index_start:index_end].cpu().numpy()
        VPD = self.VPD[index_start:index_end].cpu().numpy()
        rh = self.rh[index_start:index_end].cpu().numpy()
        gsw = self.gsw[index_start:index_end].cpu().numpy()
        return A, Q, Tleaf, VPD, rh, gsw

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

