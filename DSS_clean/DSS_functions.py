import numpy as np
from astropy.table import Table
import healpy as hp
import treecorr
from tqdm import tqdm


class DSS_class:

    def __init__(self, filter, theta_ap, nside):
    
        """ Class constructor for the filter
        Args:
            npix (_type_): number of pixel of desired aperture mass map
            theta_ap (_type_): aperture radius of desired aperture mass map (in arcmin)
            fieldsize (_type_): fieldsize of desired aperture mass map (in arcmin)
            use_polynomial_filter (bool, optional): Whether polynomial filter should be used. Defaults to False, in which case the exponential filter is used
        """
        self.theta_ap = theta_ap
        self.filter = filter
        self.nside = nside
        if(filter=='analytical'):
            self.filter_U = self.analytical_filter
        elif(filter=='top_hat'):
            self.filter_U = self.top_hat_filter
        elif(filter=='adapted'):
            self.filter_U = self.adapted_filter
        else:
            print('no valid filter choosen')
            print('Choose: adapted, top_hat or analytical')

    def analytical_filter(self,theta):
        """
        The U filter function for the aperture mass calculation from Cittenden et al. (2002)
        input: theta: aperture radius in arcmin
        """
        xsq_half = (theta/self.theta_ap)**2/2
        small_ufunc = np.exp(-xsq_half)*(1.-xsq_half)/(2*np.pi)
        return small_ufunc/self.theta_ap**2

    def top_hat_filter(self,theta):
        """
        A normal top-hat filter function that is 1 inside theta_ap and 0 otherwise
        """
        return np.where(abs(theta)<=self.theta_ap, 1, 0)


    def adapted_filter(self,theta):
        """
        The adapted filter function that can be found in filters
        """
        data=np.loadtxt('filters/adapted_filter.dat')
        filter_U = data[:,3]
        theta_up = data[:,2]
        theta_low = data[:,1]

        U = np.zeros(len(theta))
        for i in range(len(filter_U)):
            r_index = 0
            for r in theta:
                if((abs(r)<=theta_up[i])&(abs(r)>theta_low[i])):
                    U[r_index]=filter_U[i]
                if(abs(r)==0):
                    U[r_index]=filter_U[0]
                r_index+=1
        return U

    def filter_beam(self):
        """
        Get beam for healpy smoothing
        """
       
        theta = np.linspace(0,self.theta_ap*10,100000)
        bw = self.filter_U(theta=theta)
        beam = hp.sphtfunc.beam2bl(bw, np.radians(theta/60), self.nside*3)
    
        return beam,bw,theta

    def calc_Nap(self,bias,n0,density_contrast,save_fname=None):
        """
        Calculate the aperture number with bias b and galaxy number density n_0 in 1/arcmin^2
        """

        if(hp.get_nside(density_contrast)!=self.nside):
            print('density contrast does not have nisde '+str(self.nside))
            print('SGD will be scaled to nside '+str(self.nside))   
            

        n = n0*(1+bias*density_contrast)

        ### Next the surface galaxy density is created
        negative_n=np.where(n<0)[0]
        n[negative_n]=0
        SGD = np.random.poisson(n)
        pixel=np.where(SGD>0)[0]
        positive_SGD = SGD[pixel]

        SGD=hp.pixelfunc.ud_grade(SGD,nside_out=self.nside,power=-2)

        beam,filter,theta = self.filter_beam()

        pixel_area_inrad = hp.pixelfunc.nside2pixarea(nside=self.nside)
        Nap = hp.smoothing(SGD, beam_window=beam)/pixel_area_inrad

        if(save_fname!=None):
            np.save(save_fname,Nap)

        return Nap


    def calc_shear(self,Nap_table,gamma_table,N_quantiles,theta_min,theta_max,nbins):
        """
        calcualte N_quantiles shear profiles from theta_min to theta_max in nbins 
        """

        # sort Nap_table according to Nap
        Nap_table.sort('Nap')

        ### Create a source catalog, I got the correct results if gamma1 has to be negative. And specify the range where the shear profiles should be computed
        source_catalog = treecorr.Catalog(g1=gamma_table['gamma1'], g2=gamma_table['gamma2'],ra=gamma_table['ra'], dec=gamma_table['dec'],ra_units='deg', dec_units='deg')
        
        # number of pixel per quantile
        len_quantile = len(Nap_table['Nap'])//N_quantiles

        shear_table = Table()
        for quantile in tqdm(range(N_quantiles)):
            lens_catalog = treecorr.Catalog(ra=Nap_table['ra'][len_quantile*quantile:len_quantile*(quantile+1)], dec=Nap_table['dec'][len_quantile*quantile:len_quantile*(quantile+1)],ra_units='deg', dec_units='deg')

            ng = treecorr.NGCorrelation(min_sep=theta_min, max_sep=theta_max, nbins=nbins, sep_units='arcmin',bin_slop=0.02,bin_type="Log",metric='Arc')
            ng.process(lens_catalog,source_catalog)
      
            shear_table.add_column(ng.meanr,name='meanr '+str(quantile+1))
            shear_table.add_column(ng.xi,name='gamma '+str(quantile+1))
            
        return shear_table


