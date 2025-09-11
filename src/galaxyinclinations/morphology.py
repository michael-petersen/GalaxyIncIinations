from flex import FLEX
from scipy.stats import linregress
from astropy.io import fits
import numpy as np


def galaxymorphology(file):
    
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
    
    
    with fits.open(file) as hdulist:
        image_data = hdulist[1].data
        
    galaxy_name = file.split("-")[0]
    h, w = image_data.shape

    gray=image_data.copy() 
    rmaxx = h // 2
    rmaxy = w // 2
   
    radius=h//4
    cx, cy = w//2, h//2
    y, x = np.indices((h,w))
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    gray = np.where(mask, gray, 0.0)
    x_edges = np.linspace(-rmaxy, rmaxy, w + 1)
    y_edges = np.linspace(-rmaxx, rmaxx, h + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X2, Y2 = np.meshgrid(x_centers, y_centers, indexing='ij')

    R = np.sqrt(X2**2 + Y2**2).ravel()
    I = image_data.ravel()

    # Filter for positive intensities and R <= 80
    valid = (I > 0) & (R <= 80)
    R_valid = R[valid]
    logI_valid = np.log(I[valid])
   
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(R_valid, logI_valid)
    scale_length = -1 / slope
    if scale_length < 0:
        scale_length = 50
    if scale_length > 100:
        scale_length = 30
    
    a=scale_length*1.5

    R,P = np.sqrt(X2**2 + Y2**2), np.arctan2(Y2, X2)

    mmax, nmax = 8, 10


    L = FLEX(a,mmax,nmax, R.flatten(), P.flatten(), mass=image_data.flatten())
    
    c1 = L.coscoefs; s1 = L.sincoefs
   
    num = np.sqrt(sum((c1[2, n]**2 + s1[2, n]**2) for n in range(nmax)))
    den     = sum(abs(c1[0,n])            for n in range(nmax))

    eta_bt     = num/den
    A=-0.30845928737374684
    B=-1.6564105427131928
    C=6.430938401182824
    D=0.3145522357466893

    inc_bt=FindInc2(eta_bt,A,B,C,D)

    PA=90+(np.arctan2(s1[2,0],c1[2,0])* 180/np.pi)/2

    return inc_bt, PA, galaxy_name
    
