"""Masked 16-level green-channel co-occurrence features."""
import numpy as np

def texture(g,mask):
    """16-level G-channel co-occurrence, only neighboring pixels BOTH in ROI."""
    q=np.minimum(g.astype(np.int32)//16,15);m=mask>0;co=np.zeros((16,16),float)
    for a,b,valid in [(q[:,:-1],q[:,1:],m[:,:-1]&m[:,1:]),(q[:-1,:],q[1:,:],m[:-1,:]&m[1:,:])]:
        np.add.at(co,(a[valid],b[valid]),1)
    if co.sum()==0:raise ValueError('No neighboring ROI pixels')
    co+=co.T;co/=co.sum();i,j=np.indices(co.shape);positive=co>0
    return dict(G_texture_contrast=float(np.sum(co*((i-j)/15.)**2)),
                G_texture_homogeneity=float(np.sum(co/(1+abs(i-j)))),
                G_texture_energy=float(np.sum(co*co)),
                G_texture_entropy=float(-np.sum(co[positive]*np.log2(co[positive]))))
