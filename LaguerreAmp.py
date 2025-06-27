class LaguerreAmplitudes:
    """
    LaguerreAmplitudes class for calculating Laguerre basis amplitudes.
    This class provides methods for calculating Laguerre basis amplitudes based on Weinberg & Petersen (2021).
    """

    def __init__(self, rscl, mmax, nmax, R, phi, mass=1., velocity=1.):
        """
        Initialize the LaguerreAmplitudes instance with parameters.
        """
        self.rscl = rscl
        self.mmax = mmax
        self.nmax = nmax
        self.R = R
        self.phi = phi
        self.mass = mass
        self.velocity = velocity

        # Run the amplitude calculation
        self.laguerre_amplitudes()

    # Existing methods...

    def readsnapshot(self, image_array, x_edges, y_edges):
        """
        Process the image array to create pixel arrays and flatten into r, phi, snapshotflat.
        Args:
            image_array (array-like): 2D image array.
            x_edges (array-like): Bin edges for the x-axis.
            y_edges (array-like): Bin edges for the y-axis.
        Returns:
            None: Updates the instance attributes r, phi, and snapshotflat.
        """
        dx = x_edges[1] - x_edges[0]
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        xpix, ypix = np.meshgrid(x_centers, y_centers, indexing='ij')
        self.r = np.sqrt(xpix**2 + ypix**2)
        self.phi = np.arctan2(ypix, xpix)
        self.snapshotflat = image_array.flatten() * (dx * dx)

    def read(self, filename):
        """
        Read Laguerre coefficients from a file.
        Args:
            filename (str): Path to the file containing Laguerre coefficients.
        Returns:
            None: Updates the instance attributes coscoefs and sincoefs.
        """
        data = np.load(filename, allow_pickle=True)
        self.coscoefs = data['coscoefs']
        self.sincoefs = data['sincoefs']

    def save(self, filename):
        """
        Save Laguerre coefficients to a file.
        Args:
            filename (str): Path to the file where coefficients will be saved.
        Returns:
            None
        """
        np.savez(filename, coscoefs=self.coscoefs, sincoefs=self.sincoefs)