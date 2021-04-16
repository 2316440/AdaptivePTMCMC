import numpy as np
import lal
import lalsimulation as lalsim
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

def simulate_fd_waveform(m1, m2, dl, iota, phi, df, f_min, S1x=0., S1y=0., S1z=0., S2x=0., S2y=0., S2z=0., lAN=0., e=0., Ano=0.):       
        """
        Simulates frequency domain inspiral waveform

        Parameters
        ----------
        m1, m2 : float
            observed source masses in solar masses
        dl : float
            luminosity distance in Mpc
        inc: float
            source inclination in radians
        phi : float
            source reference phase in radians
        S1x, S1y, S1z : float, optional
            x,y,z-components of dimensionless spins of body 1 (default=0.)
        S2x, S2y, S2z : float, optional
            x,y,z-components of dimensionless spin of body 2 (default=0.)
        lAN: float, optional
            longitude of ascending nodes (default=0.)
        e: float, optional
            eccentricity at reference epoch (default=0.)
        Ano: float, optional
            mean anomaly at reference epoch (default=0.)
        
        Returns
        -------
        lists
            hp and hc
        """  
        approx = lalsim.IMRPhenomD #lalsim.TaylorF2
        m1, m2 = [m1*lal.MSUN_SI, m2*lal.MSUN_SI] #convert masses to SI units
        dl = dl*1e6*lal.PC_SI
        
        f_ref=40.
        f_max=0.
                      
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
                    m1, m2,
                    S1x, S1y, S1z, S2x, S2y, S2z,                  
                    dl, iota, phi, lAN, e, Ano, df, f_min, f_max, f_ref, None, approx)
       
        hp = hp.data.data     
        hc = hc.data.data
        
        #recreate frequency array
        freqs=df * np.arange(len(hp))
        
        start = int(f_min/df) #index of the first non-zero value of the fd signal (corresponding to f_min)
                          
        return hp[start:],hc[start:], freqs[start:]
    

def model(theta, detectors):
    
    #create an array to hold the frequency series of the data 
    
    freqs = detectors[0].get_frequency_array()     #get frequency array in the detector (should be same for both)
    N = len(freqs)
    f_min = freqs[0] 
    df = freqs[1]-freqs[0]
    
    signal = np.zeros(N)
    signal = signal + 0j
    
    signals = {det.name: signal for det in detectors}   #dictionary to hold model signals

    m1, m2, dl, iota, t0, psi, ra, dec = theta
    
    phi = 0. #fixed due to marginalisation
    
    hp, hc, _ = simulate_fd_waveform(m1, m2, dl, iota, phi, df, f_min)

    
    for det in detectors:
        
        #apply antenna response of the detector
        fp, fc = det.get_antenna_response(ra, dec, psi, t0)
        hf = hp*fp + hc*fc 
    
        if len(hf) > N:
            signal = hf[:N]
        else:
            signal = signals[det.name] #set signal to the 0 signal
            signal[0:len(hf)] = hf #signal padded with zeros up to Nyquist frequency
    
        #apply delay (following bibly interferometer.p, line 316-322)
        dt_geoct = t0 - det.t_start   #geocentic time of arrival - time segment start time   (time delay at geocenter, 0 for t0_true) 
        dt_delay = det.get_time_delay(ra, dec, t0)  #travel time between geocenter & detector
        tau = dt_geoct + dt_delay
    
        signal *= np.exp(-2*np.pi*1j*freqs*tau)
        signals[det.name] = signal
    
    return signals
    
    
def antenna_response(gpsTime, ra, dec, psi, det):
    """
    Get the response of a detector to plus and cross polarisation signals.
    
    Args:
        gpsTime (float): the GPS time of the observations
        ra (float): the right ascension of the source (rads)
        dec (float): the declination of the source (rads)
        psi (float): the polarisation angle of the source (rads)
        det (str): a detector name (e.g., 'H1' for the LIGO Hanford detector)
    
    Returns:
        The plus and cross response.
    """
    
    gps = lal.LIGOTimeGPS(gpsTime)
    gmst_rad = lal.GreenwichMeanSiderealTime(gps)

    # create detector-name map
    detMap = {'H1': lal.LALDetectorIndexLHODIFF,
              'H2': lal.LALDetectorIndexLHODIFF,
              'L1': lal.LALDetectorIndexLLODIFF,
              'G1': lal.LALDetectorIndexGEO600DIFF,
              'V1': lal.LALDetectorIndexVIRGODIFF,
              'T1': lal.LALDetectorIndexTAMA300DIFF}

    try:
        detector=detMap[det]
    except KeyError:
        raise ValueError("ERROR. Key {} is not a valid detector name.".format(det))

    # get detector
    detval = lal.CachedDetectors[detector]

    response = detval.response

    # actual computation of antenna factors
    fp, fc = lal.ComputeDetAMResponse(response, ra, dec, psi, gmst_rad)

    return fp, fc

def sigma_squared_noise(noise_psd, T):
    #noise psd is one sided noise psd
    return  T/2 * noise_psd

def loglikelihood(data, model, df, noise_psd):
    #data = delta t * fft of the data (one sided) 
    #model is the signal hypothesis
    #noise_psd - one sided psd of the noise
    #return -0.5 * np.sum( np.abs(data - model)**2 / sigma_noise_squared)
    
    return - 2 * df * np.sum(np.abs(data - model)**2 / noise_psd)

def likelihood(loglikelihood):
    #this is not normalized, but will be between 0-1 (1 for maximum likelihood)
    return np.exp(loglikelihood-loglikelihood.max())

def chirp_mass(m1,m2):
    return (m1*m2)**(3/5) / (m1+m2)**(1/5)

def eta(m1,m2):
    return m1*m2/(m1+m2)**2

def q(m1,m2):
    return m1/m2

def eta_M_to_m1m2(n,M):
    m1 = (M + ((1-4*n)**(1/2) * M) )/ (2*n**(3/5))
    m2 = M**2 / (n**(1/5) * m1)
    return m1,m2

def q_M_to_m1m2(q,M):
    m1=M*(1+q)**(1/5) *q**(2/5)
    m2=M*(1+q)**(1/5) *q**(-3/5)
    return m1, m2

def prior_masses_BBH(m1,m2):
    if m1 >= m2:
        return 1 
    else:
        return 0
    
def prior_q_M(q,M):
    if q < 1:
        return 0
    if M <= 0:
        return 0
    else:
        m1, m2 = q_M_to_m1m2(q,M)
        J = (M*(1+q)**(2/5)*q**(-6/5))
        #J = (M/(m2**2))
        return prior_masses_BBH(m1,m2)*J

def prior_dl(dl):
    return dl**2

def prior_dec(dec):
    return np.cos(dec)

def prior_iota(iota):
    return np.sin(iota)

def flatlogprior(theta,theta_min,theta_max):       # pass the parameter location and the parameter bounds
    if np.any(theta-theta_max>0) or np.any(theta_min-theta>0):
        return -np.inf     # if outside the bounds return probability = 0 (log-prob = -infinity)
    return 0.0             # otherwise just return log-prob = 0.0

def t_to_c(f_min, M_chirp):
#from Gravitational-wave physics and astronomy: an introduction to theory, experiment and data analysis p.324
    GM = lal.G_SI * M_chirp * lal.MSUN_SI
    c = lal.C_SI
    return 5/256 * GM / c**3 * (np.pi*GM*f_min /c**3)**(-8/3)     

def f_max(M_tot):
    #f_max = 2* f_ISO (maximum frequency of integration)       #from GW cosmo   IS THIS RIGHT??
    # M_tot - total mass of the system in M_Sun   
    M=M_tot*lal.MSUN_SI
    return 1/(6**(3/2) * np.pi * M) * lal.C_SI**3/lal.G_SI

def cosmo_distance_prior(H_0, Omega_m):
    
    Cosmo=FlatLambdaCDM(H_0, Omega_m)
        
    z=np.arange(0,1,0.0001)
    d_lum_z=np.zeros(len(z))
    for i in range(len(d_lum_z)):
        d_lum_z[i]=Cosmo.luminosity_distance(z[i]).value
    z_dl = interp1d(d_lum_z, z, kind='cubic')
        
    d_max=Cosmo.luminosity_distance(z[-1]).value
    dl = np.arange(0.,d_max,1.)
        
    cosmo_prior=np.zeros(len(dl))
    c=299792.458 #speed of light in km/s
        
    for i in range(len(dl)):
        d=dl[i]
        z=z_dl(d)
        E=Cosmo.efunc(z)
        cosmo_prior[i]=d**2 * (1+z)**(-2) * 1/(d*H_0*E/c + (1+z)**2)
        
    distance_prior_intp=interp1d(dl,cosmo_prior,kind='cubic')
    return distance_prior_intp