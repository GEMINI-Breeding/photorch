import numpy as np

def evaluateFvCB(x, p):
    """
    Evaluation of the FvCB (1980) function of photosynthetic assimilation with modified Arrhenius temperature responses.

    Parameters:
    x : numpy.ndarray
        Input array of shape (n, 3), where columns represent [Ci, Q, T] with Ci in umol/mol (or ppm), Q in umol/m2/s and T in K.
    p : pandas.Series or dict
        Parameter set with required keys: Vcmax25, Vcmax_dHa, Vcmax_dHd, Vcmax_Topt,
        Jmax25, Jmax_dHa, Jmax_dHd, Jmax_Topt, Kc25, Kc_dHa, Ko25, Ko_dHa, Gamma25,
        Gamma_dHa, Rd25, Rd_dHa, O, alpha, theta.

    Returns:
    numpy.ndarray
        Net assimilation rates for the given inputs.
    """

    # Constants
    R = 0.008314

    # Define Tresp function
    def Tresp(T, dHa, dHd, Topt):
        arrhenius = np.exp(dHa / R * (1 / 298 - 1 / T))
        f298 = 1 + np.exp(dHd / R * (1 / Topt - 1 / 298) - np.log(dHd / dHa - 1))
        fT = 1 + np.exp(dHd / R * (1 / Topt - 1 / T) - np.log(dHd / dHa - 1))
        return arrhenius * f298 / fT

    # Define temperature-dependent functions
    Vcmax = lambda T: p['Vcmax25'] * Tresp(T, p['Vcmax_dHa'], p['Vcmax_dHd'], p['Vcmax_Topt'])
    Jmax = lambda T: p['Jmax25'] * Tresp(T, p['Jmax_dHa'], p['Jmax_dHd'], p['Jmax_Topt'])
    Kc = lambda T: p['Kc25'] * Tresp(T, p['Kc_dHa'], 500, 1000)
    Ko = lambda T: p['Ko25'] * Tresp(T, p['Ko_dHa'], 500, 1000)
    Gamma = lambda T: p['Gamma25'] * Tresp(T, p['Gamma_dHa'], 500, 1000)
    Rd = lambda T: p['Rd25'] * Tresp(T, p['Rd_dHa'], 500, 1000)
    Kco = lambda T: Kc(T) * (1 + p['O'] / Ko(T))

    # Light response function J
    a = max(p['theta'], 0.0001)  # Ensure 'a' is not zero
    ia = 1 / a  # Reciprocal of a
    J = lambda Q, T: (-(-(p['alpha'] * Q + Jmax(T))) - np.sqrt((-(p['alpha'] * Q + Jmax(T)))**2 - 4 * a * (p['alpha'] * Q * Jmax(T)))) * 0.5 * ia

    # RuBisCO-limited photosynthesis
    vr = lambda Ci, T: Vcmax(T) * ((Ci - Gamma(T)) / (Ci + Kco(T))) - Rd(T)

    # Electron transport-limited photosynthesis
    jr = lambda Ci, Q, T: 0.25 * J(Q, T) * ((Ci - Gamma(T)) / (Ci + 2 * Gamma(T))) - Rd(T)

    # Smooth hyperbolic minimum of vr and jr
    hmin = lambda f1, f2: (f1 + f2 - np.sqrt((f1 + f2)**2 - 4 * 0.999 * f1 * f2)) / (2 * 0.999)

    # Net assimilation rate
    A = lambda Ci, Q, T: hmin(vr(Ci, T), jr(Ci, Q, T))

    # Inputs
    Ci = x[:, 0]
    Q = x[:, 1]
    T = x[:, 2]

    # Compute assimilation rates
    return A(Ci, Q, T)