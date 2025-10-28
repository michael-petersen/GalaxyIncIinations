from flex import FLEX

from scipy.stats import linregress
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

def FindInc2(eta, A, B, C, D):
    
    if eta>0.5:
        inc=90
    else:
        # 1) Safe inverse argument (A can be negative)
        x = np.clip((eta - D) / A, -1.0, 1.0)

        # 2) Two branches from cos symmetry, all in radians
        a = np.arccos(x)                # in [0, pi]
        i1 = (a - C) / B
        i2 = (-a - C) / B

        # 3) Convert candidates to degrees
        cand = np.rad2deg(np.array([i1, i2]))

        # 4) Reduce by the period in degrees: P = 2π/|B| (then to degrees)
        P = np.rad2deg(2*np.pi / abs(B))
        cand = cand % P                 # map into one period

        # 5) Fold into [0, 180], then reflect >90 across 90 to get [0, 90]
        cand = np.where(cand > 180.0, cand - 180.0, cand)
        cand = np.where(cand > 90.0, 180.0 - cand, cand)

        # 6) Pick the candidate that best reproduces eta (keeps you on the line)
        def model(i_deg):
            return A*np.cos(B*np.deg2rad(i_deg) + C) + D

        errs = np.abs(model(cand) - eta)
        inc=float(cand[np.argmin(errs)])
    return inc



def galaxymorphology(file,galaxy=None,data=None,noisefloor=-5.):

    if data is not None and galaxy is None:
        raise ValueError("If 'data' is provided, 'galaxy' must also be defined.")
    
    
    with fits.open(file) as hdulist:
        image_data = hdulist[1].data

        if data is not None:
            ra = data['RA_LEDA'][data['GALAXY']==galaxy]
            dec = data['DEC_LEDA'][data['GALAXY']==galaxy]

            wcs = WCS(hdulist[1].header)
            pixel_coords = wcs.world_to_pixel_values(ra, dec)
            #print("Pixel coordinates (x, y):", pixel_coords)
        
    galaxy_name = file.split("-")[0]
    h, w = image_data.shape

    # set default
    if data is None:
        cx, cy = w//2, h//2
    else:   
        cx, cy = pixel_coords[0], pixel_coords[1]

    # this choice of radius is a hyperparameter -- we may want to tune it
    radius=h//4

    y, x = np.indices((h,w))
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2

    # compute the cartesian pixel coordinates relative to center
    X2,Y2 = (x-cx), (y-cy)

    # created 1d radial profile
    R = np.sqrt(X2**2 + Y2**2).ravel()
    I = image_data.ravel()

    valid = (I > 0) & (R <= radius)
    R_valid = R[valid]
    I_valid = I[valid]

    # sort by radius for fit
    rindx = R_valid.argsort()
    R_sorted = R_valid[rindx]
    I_sorted = I_valid[rindx]

    # generate a moving average
    window_size = 100 # this is a hyperparameter
    means_R = np.convolve(R_sorted, np.ones(window_size)/window_size, mode='valid')
    means_I = np.convolve(I_sorted, np.ones(window_size)/window_size, mode='valid')
    
    # set the floor for noise
    # if this fails, fall back to half image size
    try:
        maxrad = means_R[np.where(np.log(means_I) < noisefloor)[0][0]]
    except:
        maxrad = radius
    
    valid = (I > 0) & (R <= maxrad)
    R_valid = R[valid]
    logI_valid = np.log(I[valid])
   
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(R_valid, logI_valid)
    scale_length = -1 / slope

    # we might want to check for bogus values here: but we can catch these on return as well
    
    # compute the cylindrical coordinates
    R,P = np.sqrt(X2**2 + Y2**2).flatten(), np.arctan2(Y2, X2).flatten()
    I = image_data.flatten()
    galaxy_pixels = (R <= maxrad)

    mmax, nmax = 2, 10

    afacs = np.array([0.75,1., 1.5, 2.0, 2.5,3.0])
    etalist = np.zeros([3,len(afacs)])
    palist = np.zeros(len(afacs))
    for i, afac in enumerate(afacs):
        L = FLEX(scale_length*afac,mmax,nmax, R[galaxy_pixels], P[galaxy_pixels], mass=I[galaxy_pixels])

        A2 = np.linalg.norm(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2))
        A1 = np.linalg.norm(np.sqrt(L.coscoefs[1]**2 + L.sincoefs[1]**2))
        A0 = np.linalg.norm(L.coscoefs[0])

        etalist[0,i]     = A0
        etalist[1,i]     = A1
        etalist[2,i]     = A2
        
        palist[i]      = np.nansum(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2)*np.arctan2(L.sincoefs[2], L.coscoefs[2]))/np.nansum(np.sqrt(L.coscoefs[2]**2 + L.sincoefs[2]**2))  # in radians

    return etalist, palist, scale_length, galaxy_name, radius, maxrad
    
