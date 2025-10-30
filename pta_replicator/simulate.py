"""
Code to make simulated PTA datasets with PINT
Created by Bence Becsy, Jeff Hazboun, Aaron Johnson
With code adapted from libstempo (Michele Vallisneri)

"""
import glob
import os
from dataclasses import dataclass
from astropy.time import TimeDelta
import astropy.units as u
import numpy as np

from pint.residuals import Residuals
import pint.toa as toa
from pint import models
from pint.simulation import make_fake_toas_fromMJDs
import pint.fitter
import tqdm

from enterprise.pulsar import Pulsar


@dataclass
class SimulatedPulsar:
    """
    Class to hold properties of a simulated pulsar
    """
    ephem: str = 'DE440'
    model: models.TimingModel = None
    toas: toa.TOAs = None
    residuals: Residuals = None
    name: str = None
    loc: dict = None
    added_signals: dict = None
    added_signals_time: dict = None

    def __repr__(self):
        return f"SimulatedPulsar({self.name})"

    def update_residuals(self):
        """Method to take the current TOAs and model and update the residuals with them"""
        self.residuals = Residuals(self.toas, self.model)

    def fit(self, fitter='auto', **fitter_kwargs):
        """
        Refit the timing model and update everything

        Parameters
        ----------
        fitter : str
            Type of fitter to use [auto]
        fitter_kwargs :
            Kwargs to pass onto fit_toas. Can be useful to set parameters such as max_chi2_increase, min_lambda, etc.
        """
        if fitter == 'wls':
            self.f = pint.fitter.WLSFitter(self.toas, self.model)
        elif fitter == 'gls':
            self.f = pint.fitter.GLSFitter(self.toas, self.model)
        elif fitter == 'downhill':
            self.f = pint.fitter.DownhillGLSFitter(self.toas, self.model)
        elif fitter == 'auto':
            self.f = pint.fitter.Fitter.auto(self.toas, self.model)
        else:
            err = f"{fitter=} must be one of 'wls', 'gls', 'downhill' or 'auto'"
            raise ValueError(err)
        
        self.f.fit_toas(**fitter_kwargs)
        self.model = self.f.model
        self.update_residuals()

    def write_partim(self, outpar: str, outtim: str, tempo2: bool = False):
        """Format for either PINT or Tempo2"""
        self.model.write_parfile(outpar)
        if tempo2:
            self.toas.write_TOA_file(outtim, format='Tempo2')
        else:
            self.toas.write_TOA_file(outtim)

    def update_added_signals(self, signal_name, param_dict, dt=None):
        """
        Update the timing model with a new signal
        """
        if self.added_signals is None: #this is first set to None and then to empty dict if make_ideal() is called - so good way to check if that was done
            raise ValueError("make_ideal() must be called on SimulatedPulsar before adding new signals.")
        if signal_name in self.added_signals:
            raise ValueError(f"{signal_name} already exists in the model.")
        self.added_signals[signal_name] = param_dict
        if dt is not None:
            self.added_signals_time[signal_name] = dt

    def to_enterprise(self, ephem='DE440'):
        """
        Convert to enterprise PintPulsar object
        """
        return Pulsar(self.toas, self.model, ephem=ephem, timing_package='pint')
    
    def generate_daily_avg_toas(self, ideal=False):
        """
        Compute daily averaged TOAs
        """
        
        # get a list of the systems that have observed this pulsar
        flags = list(np.unique(self.toas['f']))

        print('Pulsar {0} has {1} TOAs observed with {2} systems...'.format(self.name, len(self.toas), len(flags)))

        secperhr = 3600
        secperday = secperhr*24
        toas2 = None
        res2 = np.array([])
        
        for f in flags:
            mytoas = self.toas[self.toas['f'] == f]
            print('Filtering out {0} TOAs with flag {1} observed with {2}...'.format(len(mytoas), f,
                                                                                 mytoas['obs'][0]))
            myresiduals = np.zeros(len(mytoas))
    
            # get scaled errors
            err = self.model.scale_toa_sigma(mytoas).to(u.s).value

            # get ecorr
            U, ecorrvec = self.model.ecorr_basis_weight_pair(mytoas)
            ecorr = np.dot(U*ecorrvec, np.ones(U.shape[1]))

            avetoas, aveerr, averes = compute_daily_ave(mytoas.get_mjds().to(u.s).value,
                                                    myresiduals, err, ecorr=ecorr, dt=2*secperhr)

            if toas2 is None:
                toas2 = toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0],
                                           flags={'f': flags[0]}, freqs=np.median(mytoas['freq']),
                                           errors=aveerr*1e6, planets=True, ephem='DE440')
            else:
                toas2.merge(toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0],
                                               flags={'f': f}, freqs=np.median(mytoas['freq']),
                                               errors=aveerr*1e6, planets=True, ephem='DE440'))
            res2 = np.append(res2, averes)

        self.toas = toas2
        print('Pulsar {0} now has {1} daily averaged TOAs'.format(self.name, len(self.toas)))
        
        # remove EcorrNoise and ScaleToaError from the timing model
        self.model.remove_component('EcorrNoise')
        self.model.remove_component('ScaleToaError')
        
        if self.added_signals is None:
            self.added_signals = {}
        
        for i, flag in enumerate(flags):
            self.update_added_signals('{}_{}_measurement_noise'.format(self.name, flag),
                                     {'efac': 1.0, 'log10_t2equad': None})

        # go through and remove any maskParameters that are now empty
        empty_masks = self.model.find_empty_masks(self.toas)
        if len(empty_masks) > 0:
            for m in empty_masks:
                self.model.remove_param(m)

        # get a list of the model components
        component_names = self.model.components.keys()
    
        # remove any components that no longer have any parameters
        for name in component_names:
            if len(self.model.components[name].params) == 0:
                self.model.remove_component(name)

        # remove DMX and troposphere delay
#        if 'DispersionDMX' in component_names:
#            self.model.remove_component('DispersionDMX')
        if 'TroposphereDelay' in component_names:
            self.model.remove_component('TroposphereDelay')
        if 'FD' in component_names:
            self.model.remove_component('FD')
        
        # update residuals
        if ideal:
            make_ideal(self)
        else:
            residuals = Residuals(self.toas, self.model)
            self.toas.adjust_TOAs(TimeDelta(-1.0*residuals.time_resids + u.Quantity(res2, u.s)))
            self.update_residuals()


def simulate_pulsar(parfile: str, obstimes, toaerr, freq=1440.0, observatory="AXIS", flags=None, ephem:str = 'DE440') -> SimulatedPulsar:
    """
    Create a SimulatedPulsar object from a par file and a list of toas

    Parameters
    ----------
    parfile : str
        Path to par file
    obstimes : array
        List of observation times [MJD]
    toaerr : float or array
        Measurement error - either a common error, or a list of errors of the same length as obstimes [us]
    freq : float or array, optional
        Observation frequency - either a common value or a list [MHz]
    observatory : str, optional
        Observatory for fake toas
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError("par file does not exist.")

    model = models.get_model(parfile)
    toas = make_fake_toas_fromMJDs(obstimes, model,
                                   freq=freq * u.MHz,
                                   obs=observatory,
                                   flags=flags,
                                   error=toaerr * u.us)
    residuals = Residuals(toas, model)
    name = model.PSR.value

    if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
        loc = {'RAJ': model.RAJ.value, 'DECJ': model.DECJ.value}
    elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
        loc = {'ELONG': model.ELONG.value, 'ELAT': model.ELAT.value}
    else:
        raise AttributeError("No pulsar location information (RAJ/DECJ or ELONG/ELAT) in parfile.")
    

    return SimulatedPulsar(ephem=ephem, model=model, toas=toas, residuals=residuals, name=name, loc=loc)


def load_pulsar(parfile: str, timfile: str, ephem:str = 'DE440') -> SimulatedPulsar:
    """
    Load a SimulatedPulsar object from a par and tim file

    Parameters
    ----------
    parfile : str
        Path to par file
    timfile : str
        Path to tim file
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError("par file does not exist.")
    if not os.path.isfile(timfile):
        raise FileNotFoundError("tim file does not exist.")

    model = models.get_model(parfile)
    toas = toa.get_TOAs(timfile, ephem=ephem, planets=True)
    residuals = Residuals(toas, model)
    name = model.PSR.value

    if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
        loc = {'RAJ': model.RAJ.value, 'DECJ': model.DECJ.value}
    elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
        loc = {'ELONG': model.ELONG.value, 'ELAT': model.ELAT.value}
    else:
        raise AttributeError("No pulsar location information (RAJ/DECJ or ELONG/ELAT) in parfile.")
    

    return SimulatedPulsar(ephem=ephem, model=model, toas=toas, residuals=residuals, name=name, loc=loc)


def load_from_directories(pardir: str, timdir: str, ephem:str = 'DE440', num_psrs: int = None, debug=False) -> list:
    """
    Takes a directory of par files and a directory of tim files and
    loads them into a list of SimulatedPulsar objects
    """
    if not os.path.isdir(pardir):
        raise FileNotFoundError("par directory does not exist.")
    if not os.path.isdir(timdir):
        raise FileNotFoundError("tim directory does not exist.")
    unfiltered_pars = sorted(glob.glob(pardir + "/*.par"))
    filtered_pars = [p for p in unfiltered_pars if ".t2" not in p]
    unfiltered_tims = sorted(glob.glob(timdir + "/*.tim"))
    combo_list = list(zip(filtered_pars, unfiltered_tims))
    psrs = []
    for par, tim in combo_list:
        if num_psrs:
            if len(psrs) >= num_psrs:
                break
        if debug: print(f"loading {par=}, {tim=}")
        psrs.append(load_pulsar(par, tim, ephem=ephem))
    return psrs


def make_ideal(psr: SimulatedPulsar, iterations: int = 2):
    """
    Takes a pint.TOAs and pint.TimingModel object and effectively zeros out the residuals.
    """
    for ii in range(iterations):
        residuals = Residuals(psr.toas, psr.model)
        psr.toas.adjust_TOAs(TimeDelta(-1.0*residuals.time_resids))
    psr.added_signals = {}
    psr.added_signals_time = {}
    psr.update_residuals()


def compute_daily_ave(times, res, err, ecorr=None, dt=1.0, flags=None):
    """
    From PAL2
    ...
    Computes daily averaged residuals 
     :param times: TOAs in seconds
     :param res: Residuals in seconds
     :param err: Scaled (by EFAC and EQUAD) error bars in seconds
     :param ecorr: (optional) ECORR value for each point in s^2 [default None]
     :param dt: (optional) Time bins for averaging [default 1 s]
     :return: Average TOAs in seconds
     :return: Average error bars in seconds
     :return: Average residuals in seconds
     :return: (optional) Average flags
     """

    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(times[l]) for l in bucket_ind],'d')

    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])
    
    aveerr = np.zeros(len(bucket_ind))
    averes = np.zeros(len(bucket_ind))

    for i,l in enumerate(bucket_ind):
        M = np.ones(len(l))
        C = np.diag(err[l]**2)
        if ecorr is not None:
            C += np.ones((len(l), len(l))) * ecorr[l[0]]

        avr = 1/np.dot(M, np.dot(np.linalg.inv(C), M))
        aveerr[i] = np.sqrt(avr)
        averes[i] = avr * np.dot(M, np.dot(np.linalg.inv(C), res[l]))

    if flags is not None:
        return avetoas, aveerr, averes, aveflags
    else:
        return avetoas, aveerr, averes
