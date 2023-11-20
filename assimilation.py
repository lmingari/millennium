import numpy as np
import pandas as pd
import xarray as xr
from scipy import linalg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class AssimilationMethod:
    """
    Generic assimilation method for ensemble FALL3D
    outputs

    Attributes
    ----------
    method: str
        The assimilation method
    nens: int
        The ensemble size
    nobs: int
        Number of observation to be assimilated
    x: xarray
        Ensemble of model states
    y: xarray
        Ensemble of model states in the observation space

    Methods
    -------
    read_ensemble(fname_ens,bulk_density)
        Open an ensemble FALL3D output and convert from deposit
        mass loading in kg/m2 to deposit thickness in cm
    read_observations(fname_obs,random_sort=False)
        Read observation file with deposit thickness in cm
    apply_ObsOp()
        Apply observation operator
    to_netcdf(fname_an):
        Save analysis and forecast to a NetCDF file
    """

    methods = ['GNC','GIG','ENKF']

    def __init__(self,method):
        """
        Parameters
        ----------
        method: str
            The assimilation method
        """
        if not method in self.methods:
            str_out = "Method: {} not implemented".format(method)
            raise NotImplementedError(str_out)
        else:
            self.method = method

        self.df = None
        self.x  = None
        self.y  = None
        #
        self.xfm = None
        self.xam = None

    def read_ensemble(self,fname_ens,bulk_density):
        """
        Open an ensemble FALL3D output and convert from deposit
        mass loading in kg/m2 to deposit thickness in cm

        Parameters
        ----------
        fname_ens: str
            The ensemble FALL3D output file 
        bulk_density: float
            The deposit bulk density in kg/m3
        """
        ds = xr.open_dataset(fname_ens)
        #Convert mass loading (kg/m2) to thickness (cm)
        #using deposit bulk density (kg/m3)
        fu = 100.0/bulk_density
        self.x    = fu*ds.isel(time=-1)['tephra_grn_load']
        self.nens = self.x.sizes['ens']

    def read_observations(self,fname_obs,dataset,random_sort=False):
        """
        Read observation file with deposit thickness in cm

        Parameters
        ----------
        fname_obs: str
            Observation file
        random_sort: bool, optional
            If observations should be randomly sorted
        """
        df = pd.read_csv(fname_obs)
        df = df[['latitude','longitude',dataset,'error']].dropna()
        df.rename(columns = {dataset: 'thickness'}, inplace=True)
        if random_sort: df = df.sample(frac=1)
        self.df   = df
        self.nobs = len(df)

    def apply_ObsOp(self):
        """
        Apply observation operator
        """
        if self.x is None:
            raise TypeError("Read ensemble model state first")
        lat_obs = xr.DataArray(self.df['latitude'], dims='loc')
        lon_obs = xr.DataArray(self.df['longitude'],dims='loc')
        self.y  = self.x.interp(lat=lat_obs,lon=lon_obs)

    def to_netcdf(self,fname_an):
        """
        Save analysis and forecast to a NetCDF file

        Parameters
        ----------
        fname_an: str
            Analysis output file
        """
        ds = xr.Dataset()
        ds['forecast'] = self.xfm
        ds['analysis'] = self.xam
        ds.to_netcdf(fname_an)

    def _check_assimilate(self):
        """
        Check if required variables are defined
        """
        if self.df is None: raise TypeError("Read observations first")
        if self.y is None:  raise TypeError("Apply observation operator first")
        if self.x is None:  raise TypeError("Read ensemble model state first")

class GNC(AssimilationMethod):
    def __init__(self,max_iterations):
        super().__init__('GNC')
        self.max_iterations = max_iterations

    def assimilate(self):
        self._check_assimilate()
        y = self.y.copy()
        x = self.x.copy()
        self.xfm = x.mean(dim='ens')
        debug = True
        ###
        ### GNC method
        ###
        ym = y.mean(dim='ens')
        yp = y - ym
        yo = self.df['thickness']
        ye = self.df['error']
        
        ###
        ### Define numpy arrays for linear algebra operations
        ###
        hxm = ym.values

        if y.get_axis_num("loc") == 0:
            hx = y.values
        else:
            hx = y.values.T

        if yp.get_axis_num("loc") == 0:
            hxp = yp.values
        else:
            hxp = yp.values.T

        Ri = np.diag(1.0/ye**2)
        P  = hxp@hxp.T
        P /= self.nens-1
        #
        Pi,rank = linalg.pinvh(P,return_rank=True)
        if debug:
            print("* Checking succesful inversion:")
            print("  rank/obs: {}/{}".format(rank,self.nobs))
            result = np.allclose(P, np.dot(P, np.dot(Pi,P)))
            print("  result: {}".format(result))

        Q  = hx.T@(Ri+Pi)@hx
        b  = -1*hx.T@(Pi@hxm+Ri@yo)
        Ap = np.abs(Q)+Q
        An = np.abs(Q)-Q

        ###
        ### GNC method: solve iterative procedure
        ###
        w = np.full(self.nens,1.0/self.nens) # Weight factors: initial condition
        for i in range(self.max_iterations):
            a = Ap@w
            c = An@w
            f = (np.sqrt(b**2+a*c)-b)/a
            if(np.allclose(w*f,w)): break
            w *= f    

        if debug: print("Finishing at iteration: {}".format(i+1))
        if i+1==self.max_iterations:
            print("**WARNING** No convergence achieved")
            print("Increase the number of iterations")

        self.w = xr.DataArray(w,dims='ens')
        self.xam = x.dot(self.w)

        return w

class GIG(AssimilationMethod):
    def __init__(self,thickness_min):
        super().__init__('GIG')
        self.thickness_min = thickness_min

    def assimilate(self):
        self._check_assimilate()
        y = self.y.copy()
        x = self.x.copy()
        self.xfm = x.mean(dim='ens')
        ###
        ### GIG method (sequential form)
        ###
        for index,row in self.df.iterrows():
            yo  = row["thickness"]
            R1  = row["error_r"]**2  #type 1 relative observation error variance
            R2  = 1./(1./R1+1)       #type 2 relative observation error variance
            #
            if yo==0: yo = self.thickness_min*np.random.uniform()
            #
            yf  = y.sel(loc=index)
            yfm = yf.mean()
            P   = yf.var()
            P1  = P/yfm**2         #type 1 relative forecast error variance
            P2  = P/(P+yfm**2)     #type 2 relative forecast error variance
            #
            yam = self.__get_AnMean(yfm,yo,P2,R2)
            ya  = yam * (1 + self.__get_AnPerturbation(yf,yfm,yo,P2,R2))
            #
            dx  = xr.cov(x,yf,dim='ens') / P
            x  += dx*(ya-yf)
            x   = x.where(x>0,0)
            #
            dy  = xr.cov(y,yf,dim='ens') / P
            y  += dy*(ya-yf)
            y   = y.where(y>0,0)
        self.xam = x.mean(dim='ens').drop('loc')

    @staticmethod
    def __get_AnMean(yfm,yo,p2,r2):
        y_inv = 1.0/yfm
        y = y_inv + p2/(p2+r2)*(1./yo - (r2+1)*y_inv)
        y = 1.0 / y
        return y

    @staticmethod
    def __get_AnPerturbation(yf,yfm,yo,p2,r2):
        yg1 = r2/(1+2*r2)
        yg_mean = (1+2*r2)*yo
        yg_var  = yg1*yg_mean**2
        k = 1.0/yg1
        theta = yg_mean*yg1
        yg = np.random.gamma(k,theta,size=yf.sizes['ens'])
        #
        a_coeff = np.sqrt(1-p2)/yfm 
        b_coeff = p2/(p2+r2)
        c_coeff = 1.0/np.sqrt(yg_mean**2-2*yg_var)
        output = (yf-yfm)*a_coeff + b_coeff*(c_coeff*(yg-yg_mean) - a_coeff*(yf-yfm))
        #
        return output

class ENKF(AssimilationMethod):
    def __init__(self):
        super().__init__('ENKF')

    def assimilate(self):
        self._check_assimilate()
        y = self.y.copy()
        x = self.x.copy()
        self.xfm = x.mean(dim='ens')
        ###
        ### EnKF method
        ###
        ym = y.mean(dim='ens')
        xm = x.mean(dim='ens')
        yp = y - ym
        xp = x - xm

        y_mean    = ym.values
        x_mean    = xm.stack(z=("lat", "lon")).values
        y_perturb = yp.values.T
        x_perturb = xp.stack(z=("lat", "lon")).values.T

        yo = self.df['thickness']
        ye = self.df['error']
        R  = np.diag(ye**2)

        # Compute the Kalman gain matrix
        k = x_perturb @ y_perturb.T @ np.linalg.inv(y_perturb @ y_perturb.T + (self.nens-1)*R)

        # Update the analysis state (posterior)
        xa = x_mean + k @ (yo - y_mean)
        xa[xa<0] = 0.0
        #
        coords = xm.stack(z=("lat","lon")).coords
        self.xam = xr.DataArray(xa,coords=coords).unstack("z")

if __name__ == '__main__':
    x = ENKF()
    x.read_observations("DATA/deposit_010.csv")
    print(x.method)
