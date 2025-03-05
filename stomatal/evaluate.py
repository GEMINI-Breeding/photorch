def evaluateBMF(x, p):
    """
    Evaluation of the 2012 modified Buckley, Mott, Farquhar function for stomatal conductance.

    Parameters:
    x : numpy.ndarray
        Input array of shape (n, 2), where columns represent [Q, D], light in umol/m2/s and VPD mmol/mol, respectively.
    p : pandas.Series or dict
        Parameter set with required keys: Em, k, i0, b.

    Returns:
    numpy.ndarray
        Stomatal conductance for the given inputs.
    """
    Em = p['Em']
    i0 = p['i0']
    k = p['k']
    b = p['b']
    Q = x[:,0]
    D = x[:,1]

    return Em*(Q+i0)/(k+b*Q+(Q+i0)*D)