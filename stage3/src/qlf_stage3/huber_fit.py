"""Robust affine intensity mapping from paired root tile medians."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

def fit(x,y,method):
    if method=="raw":
        return 1.,0.
    if method=="ratio":
        return float(np.median(y/x)),0.
    sd=float(x.std())
    if sd < 1:
        return np.nan,np.nan
    center=float(x.mean())
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        try:
            m=HuberRegressor(alpha=0,epsilon=1.35,max_iter=1000,tol=1e-7)
            m.fit(((x-center)/sd).reshape(-1,1),y)
        except (ValueError,ConvergenceWarning):
            return np.nan,np.nan
    a=float(m.coef_[0]/sd)
    return (a,float(m.intercept_-a*center)) if a>0 else (np.nan,np.nan)
