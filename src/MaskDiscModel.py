import numpy as np
import time
import h5py

# for the Laguerre polynomials
from scipy.special import eval_genlaguerre

# for interpolation
from scipy import interpolate

# your LaguerreAmplitudes implementation
from src.FLEXbase import LaguerreAmplitudes


class DiscGalaxy:
    """
    DiscGalaxy: generate a flat exponential disc or ingest a real image,
    then expand into Laguerre basis, masking anything outside the galaxy radius.
    """

    def __init__(self, N=None, phasespace=None, a=1.0, M=1.0, vcirc=200.0, rmax=100.0):
        """
        Parameters:
        -----------
        N          : int or None
                     number of particles (if None, you must pass phasespace)
        phasespace : tuple of arrays (x, y, z, u, v, w)
                     initial coordinates & velocities
        a          : float
                     disc scale length
        M          : float
                     disc total mass
        vcirc      : float
                     circular velocity
        rmax       : float
                     maximum radius in units of `a`
        """
        self.a     = a
        self.M     = M
        self.vcirc = vcirc
        # physical radius limit beyond which everything is masked
        self.rmax  = rmax * self.a

        if N is not None:
            self.N = N
            self.x, self.y, self.z, self.u, self.v, self.w = self._generate_basic_disc_points()
        else:
            self.x, self.y, self.z, self.u, self.v, self.w = phasespace
            self.N = len(self.x)

    def _generate_basic_disc_points(self):
        """Generate a flat exponential disc in (x,y), z=0, with fixed vcirc."""
        # radial grid for inversion
        rgrid = np.linspace(0.0, self.rmax, 10000)

        def menclosed(r, a=self.a, m=self.M):
            return m * (1.0 - np.exp(-r/a) * (1.0 + r/a))

        m_enc = menclosed(rgrid)
        inv_cdf = interpolate.interp1d(m_enc, rgrid, bounds_error=False, fill_value=(0.0, self.rmax))

        # draw random masses in [0, M_enc(rmax)]
        np.random.seed(42)
        u = np.random.rand(self.N) * m_enc[-1]
        r = inv_cdf(u)

        # random azimuth
        phi = 2.0 * np.pi * np.random.rand(self.N)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.zeros_like(r)
        u_vel = self.vcirc * np.sin(phi)
        v_vel = self.vcirc * np.cos(phi)
        w_vel = np.zeros_like(r)

        return x, y, z, u_vel, v_vel, w_vel

    @staticmethod
    def make_rotation_matrix(xrot, yrot, zrot, euler=False):
        """Build a 3×3 rotation matrix from Tait–Bryan (or optional Euler) angles."""
        rad = np.pi / 180.0
        a, b, c = xrot*rad, yrot*rad, zrot*rad

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(a), np.sin(a)],
                       [0, -np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, -np.sin(b)],
                       [0, 1, 0],
                       [np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), np.sin(c), 0],
                       [-np.sin(c), np.cos(c), 0],
                       [0, 0, 1]])
        R = Rx @ (Ry @ Rz)

        if euler:
            # z-x-z convention
            phi, theta, psi = a, b, c
            D = np.array([[np.cos(phi),  np.sin(phi), 0],
                          [-np.sin(phi), np.cos(phi), 0],
                          [0,             0,           1]])
            C = np.array([[1, 0,            0],
                          [0, np.cos(theta), np.sin(theta)],
                          [0, -np.sin(theta), np.cos(theta)]])
            B = np.array([[np.cos(psi),  np.sin(psi), 0],
                          [-np.sin(psi), np.cos(psi), 0],
                          [0,             0,           1]])
            R = B @ (C @ D)

        return R

    def rotate_disc(self, xrotation=0.0, yrotation=0.0, zrotation=0.0, euler=False):
        """Rotate both positions and velocities by the given angles."""
        R = self.make_rotation_matrix(xrotation, yrotation, zrotation, euler=euler)

        # rotate positions
        pos = np.vstack((self.x, self.y, self.z)).T @ R
        self.x, self.y, self.z = pos[:,0], pos[:,1], pos[:,2]

        # rotate velocities
        vel = np.vstack((self.u, self.v, self.w)).T @ R
        self.u, self.v, self.w = vel[:,0], vel[:,1], vel[:,2]

    def generate_image(self, rmax, nbins, noiselevel=-1.0):
        """Bin the (x,y) into a 2D histogram and optionally add Gaussian noise."""
        img, x_edges, y_edges = np.histogram2d(
            self.x, self.y,
            bins=[nbins, nbins],
            range=[(-rmax, rmax), (-rmax, rmax)]
        )
        self.img = img.T
        self.x_edges = x_edges
        self.y_edges = y_edges
        if noiselevel > 0:
            self.noisyimage = self.img + np.random.normal(0, noiselevel, self.img.shape)
        # centers for later reconstruction
        self.x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
        self.y_centers = 0.5*(y_edges[:-1] + y_edges[1:])

    def make_expansion(self, mmax, nmax, rscl, noisy=False):
        """
        Expand the 2D image into Laguerre amplitudes, *masking outside the galaxy radius*.
        """
        # pick the snapshot (noisy vs clean)
        snapshot = self.noisyimage if noisy and hasattr(self, 'noisyimage') else self.img

        dx = self.x_edges[1] - self.x_edges[0]
        # create 2D mesh
        X, Y = np.meshgrid(self.x_centers, self.y_centers, indexing='ij')
        rr = np.sqrt(X**2 + Y**2).ravel()
        phi = np.arctan2(Y, X).ravel()
        snapshot_flat = (snapshot * dx*dx).ravel()

        # mask everything outside self.rmax
        outside = rr > self.rmax
        snapshot_flat[outside] = np.nan

        # only keep valid pixels
        valid = np.isfinite(snapshot_flat) & (snapshot_flat > 0)
        rr_mask    = rr[valid]
        phi_mask   = phi[valid]
        mass_mask  = snapshot_flat[valid]

        # call your Laguerre engine
        L = LaguerreAmplitudes(rscl, mmax, nmax, rr_mask, phi_mask, mass=mass_mask)
        # save for later
        self.r = rr_mask
        self.p = phi_mask
        return L

    def make_pointexpansion(self, mmax, nmax, rscl, noisy=False):
        """
        Expand the point cloud into Laguerre amplitudes, masking points outside self.rmax.
        """
        rr = np.sqrt(self.x**2 + self.y**2)
        phi = np.arctan2(self.y, self.x)
        mass = np.ones_like(rr) * (self.M / self.N)

        # mask out-of-disc points
        valid = rr <= self.rmax
        rr_mask   = rr[valid]
        phi_mask  = phi[valid]
        mass_mask = mass[valid]

        L = LaguerreAmplitudes(rscl, mmax, nmax, rr_mask, phi_mask, mass=mass_mask)
        self.r = rr_mask
        self.p = phi_mask
        return L

    def compute_a1(self, E):
        """
        Compute the A1 asymmetry metric from a LaguerreAmplitudes instance E.
        """
        A1 = np.linalg.norm(np.linalg.norm([E.coscoefs, E.sincoefs], axis=2)[:,1])
        A0 = np.linalg.norm(np.linalg.norm([E.coscoefs, E.sincoefs], axis=2)[:,0])
        return A1 / A0
