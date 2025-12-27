import numpy as np
from astropy import units as u
from simulate import SimulatedPulsar, make_ideal
import pint.toa as toa
from pint.residuals import Residuals
from astropy.time import TimeDelta

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


def generate_daily_avg_toas(pulsar: SimulatedPulsar, ideal=False):
    """
    Compute daily averaged TOAs
    """
    
    # get a list of the systems that have observed this pulsar
    flags = list(np.unique(pulsar.toas['f']))

    print('Pulsar {0} has {1} TOAs observed with {2} systems...'.format(pulsar.name, len(pulsar.toas), len(flags)))

    secperday = 3600*24
    toas2 = None
    res2 = np.array([])
    
    for f in flags:
        mytoas = pulsar.toas[pulsar.toas['f'] == f]
        print('Filtering out {0} TOAs with flag {1} observed with {2}...'.format(len(mytoas), f,
                                                                                mytoas['obs'][0]))
        myresiduals = np.zeros(len(mytoas))

        # get scaled errors
        err = pulsar.model.scale_toa_sigma(mytoas).to(u.s).value

        # get ecorr
        U, ecorrvec = pulsar.model.ecorr_basis_weight_pair(mytoas)
        ecorr = np.dot(U*ecorrvec, np.ones(U.shape[1]))

        avetoas, aveerr, averes = compute_daily_ave(mytoas.get_mjds().to(u.s).value,
                                                myresiduals, err, ecorr=ecorr, dt=secperday)

        if toas2 is None:
            toas2 = toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0], flags={'f': flags[0]},
                                errors=aveerr*1e6, planets=True, ephem='DE440')
        else:
            toas2.merge(toa.get_TOAs_array(avetoas/secperday, obs=mytoas['obs'][0], flags={'f': f},
                                    errors=aveerr*1e6, planets=True, ephem='DE440'))
        res2 = np.append(res2, averes)

    pulsar.toas = toas2
    print('Pulsar {0} now has {1} daily averaged TOAs'.format(pulsar.name, len(pulsar.toas)))
    
    # remove EcorrNoise and ScaleToaError from the timing model
    pulsar.model.remove_component('EcorrNoise')
    pulsar.model.remove_component('ScaleToaError')
    
    if pulsar.added_signals is None:
        pulsar.added_signals = {}
    
    for i, flag in enumerate(flags):
        pulsar.update_added_signals('{}_{}_measurement_noise'.format(pulsar.name, flag),
                                    {'efac': 1.0, 'log10_t2equad': None})

    # go through and remove any maskParameters that are now empty
    empty_masks = pulsar.model.find_empty_masks(pulsar.toas)
    if len(empty_masks) > 0:
        for m in empty_masks:
            pulsar.model.remove_param(m)

    # get a list of the model components
    component_names = pulsar.model.components.keys()

    # remove any components that no longer have any parameters
    for name in component_names:
        if len(pulsar.model.components[name].params) == 0:
            pulsar.model.remove_component(name)

    # remove DMX and troposphere delay
    if 'DispersionDMX' in component_names:
        pulsar.model.remove_component('DispersionDMX')
    if 'TroposphereDelay' in component_names:
        pulsar.model.remove_component('TroposphereDelay')
    if 'FD' in component_names:
        pulsar.model.remove_component('FD')
    
    # update residuals
    if ideal:
        make_ideal(pulsar)
    else:
        residuals = Residuals(pulsar.toas, pulsar.model)
        pulsar.toas.adjust_TOAs(TimeDelta(-1.0*residuals.time_resids + u.Quantity(res2, u.s)))
        pulsar.update_residuals()
