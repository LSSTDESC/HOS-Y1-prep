"""Functions and classes for the extraction of aperture mass statistics from simulations using the FFT-method
authors: Laila Linke & Sven Heydenreich
"""

import numpy as np
from scipy.interpolate import griddata
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel, convolve_fft
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import collections
import multiprocessing.managers
from multiprocessing import Pool
from tqdm import tqdm
import healpy as hp

def measureMap3FromKappa_crossbins(kappa1, kappa2, kappa3, thetas=[4,8,16,32], nside=2048, verbose=False, doPlots=False, fn_out=""):
    """ Measure tomographic <Map3> from three convergence maps, given as Healpy maps

    Args:
        kappa1 (Healpy map): Convergence kappa of redshift z1. Important: Masked value need to be set to healpy.pixelfunc.UNSEEN
        kappa2 (Healpy map): Convergence kappa of redshift z2. Important: Masked value need to be set to healpy.pixelfunc.UNSEEN
        kappa3 (Healpy map): Convergence kappa of redshift z3. Important: Masked value need to be set to healpy.pixelfunc.UNSEEN
        thetas (list, optional): List of aperture scale radii in arcmin. Defaults to [4,8,16,32].
        nside (int, optional): Wished nside for convergence and aperture mass maps. Kappa maps are degraded to this value. Defaults to 2048.
        verbose (bool, optional): To switch on verbose mode. Defaults to False.
        doPlots (bool, optional): To switch on intermediate plot creation. Only makes sense when using this function inside a jupyter notebook. Defaults to False.
        fn_out (str, optional): Outputfilename for <Map3>. Defaults to "", in which case no output is made.
    """

    Nthetas=len(thetas) # Number of aperture radii

    # Downgrade kappa map to specified nside
    if(verbose):
        print(f"Downgrading kappa1 map from nside={hp.pixelfunc.get_nside(kappa1)} to nside={nside}")
    kappa1=hp.ud_grade(kappa1, nside)
    if(verbose):
        print(f"Downgrading kappa2 map from nside={hp.pixelfunc.get_nside(kappa2)} to nside={nside}")
    kappa2=hp.ud_grade(kappa2, nside)
    if(verbose):
        print(f"Downgrading kappa3 map from nside={hp.pixelfunc.get_nside(kappa3)} to nside={nside}")
    kappa3=hp.ud_grade(kappa3, nside)


    # Set mask
    mask1=np.where(kappa1!=hp.pixelfunc.UNSEEN)
    mask2=np.where(kappa2!=hp.pixelfunc.UNSEEN)
    mask3=np.where(kappa3!=hp.pixelfunc.UNSEEN)
    
    # Calculate aperture mass maps
    Maps_z1=[] # list that will contain aperture mass maps of zbin1
    Maps_z2=[] # list that will contain aperture mass maps of zbin2
    Maps_z3=[] # list that will contain aperture mass maps of zbin3
    for i,theta in enumerate(thetas):
        if verbose:
            print(f"Calculating aperture mass map for theta {theta}, ({i+1}/{Nthetas})")
        ac=aperture_mass_computer_curved_sky(nside, theta) # Calculation object for aperture mass map calculation
        map_z1=ac.Map_fft_from_kappa(kappa1) # Calculate aperture mass map with FFT for zbin1
        map_z2=ac.Map_fft_from_kappa(kappa2) # Calculate aperture mass map with FFT for zbin1
        map_z3=ac.Map_fft_from_kappa(kappa3) # Calculate aperture mass map with FFT for zbin1
        if doPlots:
            hp.mollview(map_z1, title=r"$M_\mathrm{ap}$ for $\theta$="+f"{theta} arcmin and zbin1")
        Maps_z1.append(map_z1) # Add aperture mass map to list
        Maps_z2.append(map_z2) # Add aperture mass map to list
        Maps_z3.append(map_z3) # Add aperture mass map to list

    # Calculate Map³
    if verbose:
        print(r"Calculating $M_\mathrm{ap}^3$")

    Map3=np.zeros(Nthetas*(Nthetas+1)*(Nthetas+2)//6) #Vector for Map3 measurement
    counter=0
    for i in range(Nthetas):
        for j in range(i, Nthetas):
            for k in range(j, Nthetas):
                map3_mean=np.mean(Maps_z1[i][mask1]*Maps_z2[j][mask2]*Maps_z3[k][mask3])
                Map3[counter]=map3_mean
                counter+=1
    
    if(fn_out!=""): #Save if outputfilename is given
        np.savetxt(fn_out, Map3)

    if doPlots:
        initPlot(usetex=False)
        fig, ax=plt.subplots()
        prepareMap3Plot(ax)
        ax.plot(Map3)
        ax.set_ylabel(r"$\langle M_\mathrm{ap}^3 \rangle$")
        finalizePlot(ax, title=r"$\langle M_\mathrm{ap}^3\rangle$ from curved sky "+f"nside={nside}")
        
        
def measureMap3FromKappa(kappa, thetas=[4,8,16,32], nside=2048, verbose=False, doPlots=False, fn_out=""):
    """ Measured <Map3> from a convergence map, given as Healpy map

    Args:
        kappa (Healpy map): Convergence kappa. Important: Masked value need to be set to healpy.pixelfunc.UNSEEN
        thetas (list, optional): List of aperture scale radii in arcmin. Defaults to [4,8,16,32].
        nside (int, optional): Wished nside for convergence and aperture mass map. Kappa map is degraded to this value. Defaults to 2048.
        verbose (bool, optional): To switch on verbose mode. Defaults to False.
        doPlots (bool, optional): To switch on intermediate plot creation. Only makes sense when using this function inside a jupyter notebook. Defaults to False.
        fn_out (str, optional): Outputfilename for <Map3>. Defaults to "", in which case no output is made.
    """

    Nthetas=len(thetas) # Number of aperture radii

    # Downgrade kappa map to specified nside
    if(verbose):
        print(f"Downgrading kappa map from nside={hp.pixelfunc.get_nside(kappa)} to nside={nside}")
    kappa=hp.ud_grade(kappa, nside)


    # Set mask
    mask=np.where(kappa!=hp.pixelfunc.UNSEEN)
    
    # Calculate aperture mass maps
    Maps=[] # list that will contain aperture mass maps
    for i,theta in enumerate(thetas):
        if verbose:
            print(f"Calculating aperture mass map for theta {theta}, ({i+1}/{Nthetas})")
        ac=aperture_mass_computer_curved_sky(nside, theta) # Calculation object for aperture mass map calculation
        map=ac.Map_fft_from_kappa(kappa) # Calculate aperture mass map with FFT
        if doPlots:
            hp.mollview(map, title=r"$M_\mathrm{ap}$ for $\theta$="+f"{theta} arcmin")
        Maps.append(map) # Add aperture mass map to list

    # Calculate Map³
    if verbose:
        print(r"Calculating $M_\mathrm{ap}^3$")

    Map3=np.zeros(Nthetas*(Nthetas+1)*(Nthetas+2)//6) #Vector for Map3 measurement
    counter=0
    for i in range(Nthetas):
        for j in range(i, Nthetas):
            for k in range(j, Nthetas):
                map3_mean=np.mean(Maps[i][mask]*Maps[j][mask]*Maps[k][mask])
                Map3[counter]=map3_mean
                counter+=1
    
    if(fn_out!=""): #Save if outputfilename is given
        np.savetxt(fn_out, Map3)

    if doPlots:
        initPlot(usetex=False)
        fig, ax=plt.subplots()
        prepareMap3Plot(ax)
        ax.plot(Map3)
        ax.set_ylabel(r"$\langle M_\mathrm{ap}^3 \rangle$")
        finalizePlot(ax, title=r"$\langle M_\mathrm{ap}^3\rangle$ from curved sky "+f"nside={nside}")


class aperture_mass_computer:
    """
    a class handling the computation of aperture masses for flat sky maps.
    The class can use both the exponential and the polynomial filter, but most tests were done for the exponential filter!
    initialization:
        npix: number of pixel of desired aperture mass map
        theta_ap: aperture radius of desired aperture mass map (in arcmin)
        fieldsize: fieldsize of desired aperture mass map (in arcmin)
    """

    def __init__(self, npix, theta_ap, fieldsize, use_polynomial_filter=False):
        """ Class constructor

        Args:
            npix (_type_): number of pixel of desired aperture mass map
            theta_ap (_type_): aperture radius of desired aperture mass map (in arcmin)
            fieldsize (_type_): fieldsize of desired aperture mass map (in arcmin)
            use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, in which case the exponential filter is used
        """
        self.theta_ap = theta_ap
        self.npix = npix
        self.fieldsize = fieldsize
        self.use_polynomial_filter = use_polynomial_filter
        if(use_polynomial_filter):
            print("WARNING! Using polynomial filter!")

        # compute distances to the center in arcmin
        idx, idy = np.indices([self.npix, self.npix])
        idx = idx - ((self.npix)/2)
        idy = idy - ((self.npix)/2)

        self.idc = idx + 1.0j*idy
        self.dist = np.abs(self.idc)*self.fieldsize/self.npix

        # compute the Q filter function on a grid
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()

        self.disk = np.zeros((self.npix, self.npix))
        self.disk[(self.dist < self.theta_ap)] = 1

    def change_theta_ap(self, theta_ap):
        """ Changes the aperture radius and recomputes Q and U for the pixel grid

        Args:
            theta_ap (_type_): new aperture radius [arcmin]
        """
        self.theta_ap = theta_ap
        self.q_arr = self.Qfunc_array()
        self.u_arr = self.Ufunc_array()
        self.disk = np.zeros((self.npix, self.npix))
        self.disk[(self.dist < self.theta_ap)] = 1

    def Ufunc(self, theta):
        """
        The U filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: U [arcmin^-2]
        """
        xsq_half = (theta/self.theta_ap)**2/2
        small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
        return small_ufunc/self.theta_ap**2

    def Qfunc(self, theta):
        """
        The Q filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: Q [arcmin^-2]
        """
        thsq = (theta/self.theta_ap)**2
        if(self.use_polynomial_filter):
            res = 6/np.pi*thsq**2*(1.-thsq**2)
            res[(thsq > 1)] = 0
            return res/self.theta_ap**2
        else:
            res = thsq/(4*np.pi*self.theta_ap**2)*np.exp(-thsq/2)
            return res

    def Qfunc_array(self):
        """
        Computes the Q filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            res = self.Qfunc(self.dist)*(np.conj(self.idc)
                                         ** 2/np.abs(self.idc)**2)
        res[(self.dist == 0)] = 0
        return res

    def Ufunc_array(self):
        """
        Computes the U filter function on an npix^2 grid
        fieldsize: size of the grid in arcmin
        """
        res = self.Ufunc(self.dist)
        return res

    def interpolate_nans(self, array, interpolation_method, fill_value):
        """
        method to interpolate nans. adapted from
        https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        This is needed if there are holes in the galaxy data
        """

        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        # mask invalid values
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        # get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        GD1 = griddata((x1, y1), newarr.ravel(),
                       (xx[array.mask], yy[array.mask]),
                       method=interpolation_method,
                       fill_value=fill_value)

        array[array.mask] = GD1
        # 'cubic' interpolation would probalby be better, but appears to be extremely slow
        return array

    def filter_nans_astropy(self, array):
        """Interpolate nans using the built-in functions of astropy and a Gaussian Kernel

        Args:
            array (_type_): array with nans that need to be interpolated over
        """
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize

        kernel = Gaussian2DKernel(x_stddev=filter_radius)
        filtered_array_real = interpolate_replace_nans(
            array.real, kernel, convolve=convolve_fft, allow_huge=True)
        filtered_array_imag = interpolate_replace_nans(
            array.imag, kernel, convolve=convolve_fft, allow_huge=True)

        return filtered_array_real + 1.0j*filtered_array_imag

    def filter_nans_gaussian(self, array):
        """Interpolate nans using a Gaussian filter

        Args:
            array (_type_): array with nans that need to be interpolated over
        """
        filter_radius = 0.1*self.theta_ap * self.npix/self.fieldsize
        mask = np.isnan(array)
        array[mask] = 0

        # fill an array with ones wherever there is data
        normalisation = np.ones(array.shape)
        normalisation[mask] = 0
        filtered_array_real = gaussian_filter(array.real, filter_radius)
        filtered_array_imag = gaussian_filter(array.imag, filter_radius)

        filtered_normalisation = gaussian_filter(normalisation, filter_radius)

        result = (filtered_array_real + 1.0j*filtered_array_imag) / \
            filtered_normalisation

        array[mask] = result[mask]

        return array

    def Map_fft_from_kappa(self, kappa_arr):
        """ Calculates the Aperture Mass map from a kappa grid using FFT

        Args:
            kappa_arr (_type_): Kappa grid with npix^2 values

        """
        # If U is not yet calculated, calculate U
        if self.u_arr is None:
            self.u_arr = self.Ufunc_array()

        # Do the calculation, the normalisation is the pixel size
        return fftconvolve(kappa_arr, self.u_arr, 'same')*self.fieldsize**2/self.npix**2

    def Map_fft(self, gamma_arr, norm=None, return_mcross=False, normalize_weighted=True, periodic_boundary=True):
        """
        Computes the Aperture Mass map from a Gamma-Grid
        input:
            gamma_arr: npix^2 grid with sum of ellipticities of galaxies as (complex) pixel values
            norm: if None, assumes that the gamma_arr is a field with <\gamma_t> as pixel values
                    if Scalar, uses the Estimator of Bartelmann & Schneider (2001) with n as the scalar
                    if array, computes the mean number density within an aperture and uses this for n in
                                the Bartelmann & Schneider (2001) Estimator
            return_mcross: bool -- if true, also computes the cross-aperture map and returns it
            periodic_boundary: bool (default:False) 
                               if true: computes FFT with astropy's convolve_fft without zero-padding, 
                               if false: computes FFT with scipy's fftconvolve which uses zero padding 
        output:
            result: resulting aperture mass map and, if return_mcross, the cross aperture map
        this uses Map(theta) = - int d^2 theta' gamma(theta) Q(|theta'-theta|)conj(theta'-theta)^2/abs(theta'-theta)^2
        """

        yr = gamma_arr.real
        yi = gamma_arr.imag
        qr = self.q_arr.real
        qi = self.q_arr.imag

        if periodic_boundary:  # No Zeropadding
            rr = convolve_fft(yr, qr, boundary='wrap', normalize_kernel=False,
                              nan_treatment='fill', allow_huge=True)
            ii = convolve_fft(yi, qi, boundary='wrap', normalize_kernel=False,
                              nan_treatment='fill', allow_huge=True)
        else:  # With Zeropadding
            rr = fftconvolve(yr, qr, 'same')
            ii = fftconvolve(yi, qi, 'same')

        result = (ii-rr)
        if(np.any(np.isnan(result))):
            print("ERROR! NAN in aperture mass computation!")
        if(return_mcross):
            if periodic_boundary:
                ri = convolve_fft(
                    yr, qi, boundary='wrap', normalize_kernel=False, nan_treatment='fill', allow_huge=True)
                ir = convolve_fft(
                    yi, qr,  boundary='wrap', normalize_kernel=False, nan_treatment='fill', allow_huge=True)
            else:
                ri = fftconvolve(yr, qi, 'same')
                ir = fftconvolve(yi, qr, 'same')
            mcross = (-ri - ir)

        if norm is None:
            result *= self.fieldsize**2/self.npix**2
            if(return_mcross):
                mcross *= self.fieldsize**2/self.npix**2
                return result, mcross
            return result

        if(normalize_weighted):
            if not norm.shape == gamma_arr.shape:
                print("Error! Wrong norm format")
                return None
            norm_weight = self.norm_fft(norm)
            result /= (norm_weight)
            if(return_mcross):
                mcross /= (norm_weight)
                return result, mcross
            return result

        elif isinstance(norm, (collections.Sequence, np.ndarray)):
            mean_number_within_aperture = fftconvolve(norm, self.disk, 'same')
            mean_number_density_within_aperture = mean_number_within_aperture / \
                (np.pi*self.theta_ap**2)
            result /= mean_number_density_within_aperture
            if(return_mcross):
                mcross /= mean_number_density_within_aperture
                return result, mcross
            return result

        else:
            result *= self.fieldsize**2 / norm
            if(return_mcross):
                mcross *= self.fieldsize**2 / norm
                return result, mcross
            return result

    def norm_fft(self, norm):
        q = np.abs(self.q_arr)
        result = fftconvolve(norm, q, 'same')
        return result

    def normalize_shear(self, Xs, Ys, shears, CIC=True, normalize=False, nan_treatment=None, fill_value=0, debug=False):
        """
        distributes a galaxy catalogue on a pixel grid
        input:
            Xs: x-positions (arcmin)
            Ys: y-positions (arcmin)
            shears: measured shear_1 + 1.0j * measured shear_2
            CIC: perform a cloud-in-cell interpolation
            debug: output different stages of the CIC interpolation
        output:
            zahler_arr: npix^2 grid of sum of galaxy ellipticities
        """
        npix = self.npix
        fieldsize = self.fieldsize
        if not CIC:
            shears_grid_real = np.histogram2d(Xs, Ys, bins=np.arange(
                npix+1)/npix*fieldsize, weights=shears.real)[0]
            shears_grid_imag = np.histogram2d(Xs, Ys, bins=np.arange(
                npix+1)/npix*fieldsize, weights=shears.imag)[0]
            norm = np.histogram2d(
                Xs, Ys, bins=np.arange(npix+1)/npix*fieldsize)[0]

        else:
            cell_size = fieldsize/(npix-1)

            index_x = np.floor(Xs/cell_size)
            index_y = np.floor(Ys/cell_size)

            difference_x = (Xs/cell_size-index_x)
            difference_y = (Ys/cell_size-index_y)

            hist_bins = np.arange(npix+1)/(npix-1)*(fieldsize)

            # lower left
            shears_grid_real = np.histogram2d(Xs, Ys, bins=hist_bins,
                                              weights=shears.real*(1-difference_x)*(1-difference_y))[0]
            shears_grid_imag = np.histogram2d(Xs, Ys, bins=hist_bins,
                                              weights=shears.imag*(1-difference_x)*(1-difference_y))[0]
            norm = np.histogram2d(Xs, Ys, bins=hist_bins,
                                  weights=(1-difference_x)*(1-difference_y))[0]

            # lower right
            shears_grid_real += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                               weights=shears.real*(difference_x)*(1-difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                               weights=shears.imag*(difference_x)*(1-difference_y))[0]
            norm += np.histogram2d(Xs+cell_size, Ys, bins=hist_bins,
                                   weights=(difference_x)*(1-difference_y))[0]

            # upper left
            shears_grid_real += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                               weights=shears.real*(1-difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                               weights=shears.imag*(1-difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs, Ys+cell_size, bins=hist_bins,
                                   weights=(1-difference_x)*(difference_y))[0]

            # upper right
            shears_grid_real += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                               weights=shears.real*(difference_x)*(difference_y))[0]
            shears_grid_imag += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                               weights=shears.imag*(difference_x)*(difference_y))[0]
            norm += np.histogram2d(Xs+cell_size, Ys+cell_size, bins=hist_bins,
                                   weights=(difference_x)*(difference_y))[0]

        result = (shears_grid_real + 1.0j*shears_grid_imag)

        if not normalize:
            return result, norm

        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result /= norm

        # treat the nans
        if(nan_treatment in ['linear', 'cubic', 'nearest']):
            result = self.interpolate_nans(result, nan_treatment, fill_value)
        elif (nan_treatment == 'fill'):
            result[np.isnan(result)] = fill_value
        elif (nan_treatment == 'gaussian'):
            result = self.filter_nans_gaussian(result)
        elif (nan_treatment == 'astropy'):
            result = self.filter_nans_astropy(result)

        return result


class aperture_mass_computer_curved_sky:
    """
    a class handling the computation of aperture masses for healpy curved maps.
    Currently uses only the exponential filter. Currently only accepts kappa-maps (not shear catalogues)
    initialization:
        nside: nside of healpy maps
        theta_ap: aperture radius of desired aperture mass map (in arcmin)
        beam_U: U-function calculated on a healpy "beam"
        npix: number of pixels corresponding to nside
    """

    def __init__(self, nside, theta_ap):
        """Class constructor

        Args:
            nside (float): nside of healpy maps
            theta_ap (float): aperture radius of desired aperture mass map [arcmin]
        """
        self.nside=nside
        self.theta_ap=theta_ap
        self.beam_U=self.Ufunc_beam()
        self.npix=hp.nside2npix(nside)

        # # I am not entirely sure what these things do 
        # patch_field=np.zeros(self.npix)

        # self.patch_indices = np.arange(0,self.npix)
        # patch_field[self.patch_indices]=self.patch_indices+1
        # patch_field_highres = hp.ud_grade(patch_field,nside_out=2**12)
        # patch_pixel = []
        # for i in self.patch_indices:
        #     patch_pixel.append(np.where(patch_field_highres==i+1)[0])
        # self.patch_pixel = np.array(patch_pixel)



    def change_theta_ap(self, theta_ap):
        """Changes the aperture radius and recomputes the U-beam

        Args:
            theta_ap (_type_): new aperture radius [arcmin]
        """
        self.theta_ap=theta_ap
        self.beam_U=self.Ufunc_beam()

    
    def Ufunc(self, theta):
        """
        The U filter function for the aperture mass calculation from Schneider et al. (2002)
        input: theta: aperture radius in arcmin
        output: U [arcmin^-2]
        """
        xsq_half = (theta/self.theta_ap)**2/2
        small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
        return small_ufunc/self.theta_ap**2

    def Ufunc_beam(self):
        """
        Calculates the U filter function on a healpy "beam"
        """
        b_rad=np.linspace(0, np.radians(500), 100000) #Calculate U up to 500 deg
        b_arcmin=b_rad*3438
        bw_arcmin=self.Ufunc(b_arcmin)
        bw_rad=bw_arcmin*3438*3438
        beam=hp.beam2bl(bw_rad, b_rad, self.nside*3)
        return beam

    
    def Map_fft_from_kappa(self, kappamap):
        """Calculate the aperture mass map from a kappa map using FFT

        Args:
            kappamap (Healpy map): Kappa healpy map with nside=self.nside
        """
        if self.beam_U is None:
            self.beam_U=self.Ufunc_beam()

        Map=hp.smoothing(kappamap, beam_window=self.beam_U, verbose=False)
        return Map   

