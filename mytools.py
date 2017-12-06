import numpy as np
from find_nearest import find_nearest
from pycurrents.num import eof as pycur_eof 
from pycurrents.num import Runstats
from scipy.signal import detrend as sci_detrend
from scipy.signal import windows as windows
from scipy.ndimage import filters
from butter_bandpass_filter import butter_bandpass_filter as bandpass
from scipy.signal import convolve2d
from scipy.signal import fftconvolve as sci_fftconvolve, convolve as sci_convolve
from scipy.signal import butter, lfilter, buttord, filtfilt, firwin
from scipy import stats
#import kPyWavelet as wavelet
from pylab import find, figure, axes, draw, show
from matplotlib import pyplot as plt
from scipy.interpolate import griddata as griddata
from scipy import mgrid as mgrid

########################################################################
# Utility functions
########################################################################
def masked_interp(t, y):
    """
    gap filling with linear interolation for masked arrays
    loops over 2nd dim and interps masked indices of 1st dim
    """
    yn = y.data.copy().astype(t.dtype)

    for n in range(0, y.shape[1]):
        yn[y[:, n].mask, n] = np.interp(t[y[:, n].mask], t[~y[:, n].mask], y[:, n].compressed())

    return yn

def zero_pad(y, ax = 0):
    """
    pads the beguining and end of y along the axis ax with zeros
    """

    N = y.shape[ax]
    N2 = 2**(np.ceil(np.log2(N)))
    Npad = np.ceil(.5 * (N2 - N))
    if 2*Npad + N > N2:
        N2 = 2**(np.ceil(np.log2(N+2*Npad)))
        Npad = np.ceil(.5 * (N2 - N))
    if ax == 0 and y.ndim == 1:
        pads = np.zeros((Npad,))
    elif ax == 0 and y.ndim >= 2:
        pads = np.zeros((Npad,) + y.shape[1:])
    elif ax == 1 and y.ndim ==2:
        pads = np.zeros((len(y), Npad))
    elif ax == 1 and y.ndim >=3:
        pads = np.zeros((len(y), Npad) + y.shape[2:])
    elif ax == 2 and y.ndim ==3:
        pads = np.zeros((len(y), y.shape[1], Npad))
    elif ax == 2 and y.ndim == 4:
        pads = np.zeros((len(y), y.shape[1], Npad) + y.shape[3:])
    else:
        raise ValueError, "Too many dimensions to pad or wrong axis choice."
    
    yn = np.concatenate((pads, y, pads), axis = ax)
    return yn

def freq_spec(x, y1, y2 = None, win = None, pad = True):
    """
    Compute frequency rotary spectrum: time must be 1st index
    if you want the one sided: take the neg freqs part and *2 as G = 2*S[freqs<=0]
    NOTE: if you care about the amount of variance in a given frequency, DO NOT zero pad
    if you window (but no padding) you may also need to do further bandwith corretion
    """
    x = np.asarray(x)
    if y2 == None:
        y = np.asarray(y1)
    else:
        y = np.asarray(y1 + 1j*y2)
    
    if pad == True:
        y = zero_pad(y, ax = 0)
    
    N = len(y)
    # d = (np.diff(x, axis=0)).mean(axis = 0)
    d = (np.diff(x, axis=0)).mean()
    #d = (x[1] - x[0])
    if not win in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
        raise ValueError, "Window choice is invalid"
    
    if win != None:
        win = eval('windows.' + win + '(N)')
        dofw = len(win) / np.sum(win**2)
        win.resize((N,) + tuple(np.int8(np.ones(y1.ndim - 1))))
        y = y * win
        # y = sci_detrend(y, axis =0)
    else:
        dofw = 1.0
    fy = np.fft.fftshift(np.sqrt(dofw)*np.fft.fft(y, axis=0), axes=(0,))
    py = (d/N)*np.abs(fy)**2
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d))
    if N % 2 == 1:
        epsp = np.arctan(fy[freqs >= 0].imag/ np.flipud(fy[freqs <= 0].real))
        epsn = np.arctan(np.flipud(fy[freqs <= 0].imag)/ fy[freqs >= 0].real)
        eps = np.fft.fftshift(np.concatenate((epsp, epsn[-1:0:-1]), axis=0), axes=0)
        theta = .5*(epsp + epsn)
        theta = np.fft.fftshift(np.concatenate((theta, theta[-1:0:-1]), axis=0), axes=0)
        rf = (py[freqs>=0]-np.flipud(py[freqs<=0]))/(py[freqs>=0]+np.flipud(py[freqs<=0]))
    else:
        epsp = np.arctan(fy[freqs >= 0].imag/ np.flipud(fy[(freqs <= 0) & (freqs != freqs[0])].real))
        epsn = np.arctan(np.flipud(fy[(freqs <= 0) & (freqs != freqs[0])].imag)/ fy[freqs >= 0].real)
        eps = np.fft.fftshift(np.concatenate((epsp, epsn[-1:0:-1]), axis=0), axes=0)
        theta = .5*(epsp + epsn)
        theta = np.fft.fftshift(np.concatenate((theta, theta[-1:0:-1]), axis=0), axes=0)
        rf = (py[freqs>=0]-np.flipud(py[(freqs <= 0) & (freqs != freqs[0])]))/(py[freqs>=0]+np.flipud(py[(freqs <= 0) & (freqs != freqs[0])]))
    
    
    return py, freqs, eps, theta, rf

def wavenum_spec(x, y1, y2 = None):
    """
    Compute wavenumber rotary spectrum: depth must be 2nd index: THIS FUNC NEEDS REFORM see above
    """
    x = np.asarray(x)
    if y2 == None:
        y = np.asarray(y1)
    else:
        y = np.asarray(y1 + 1j*y2)
    
    N = len(x)
    d = (np.diff(x)).mean(axis = 0)
    #d = (x[1] - x[0])
    fy = np.fft.fftshift(np.fft.fft.fft(y, axis=1), axes=(1,))
    py = (d/N)*np.abs(fy)**2
    wavn = np.fft.fftshift(np.fft.fftfreq(N, d))
    if N % 2 == 1:
        epsp = np.arctan(fy[:, wavn >= 0].imag/np.fliplr(fy[:, wavn <= 0].real))
        epsn = np.arctan(np.fliplr(fy[:, wavn <= 0].imag)/fy[:, wavn >= 0].real)
        theta = .5*(epsp + epsn)
        theta = np.fft.fftshift(np.concatenate((theta, theta[:, -1:0:-1]), axis=1), axes=1)
        rf = (py[:, wavn>=0] - np.fliplr(py[:, wavn<=0])) / (py[:, wavn>=0] + np.fliplr(py[:, wavn<=0]))
    else:
        epsp = np.arctan(fy[:, wavn >= 0].imag / np.fliplr(fy[:, wavn[1:] <= 0].real))
        epsn = np.arctan(np.fliplr(fy[:, wavn[1:] <= 0].imag) / fy[:, wavn >= 0].real)
        theta = .5*(epsp + epsn)
        theta = np.fft.fftshift(np.concatenate((theta, theta[:, -1:0:-1]), axis=1), axes=1)
        rf = (py[:, wavn>=0] - np.fliplr(py[:, wavn[1:]<=0])) / (py[:, wavn>=0] + np.fliplr(py[:, wavn[1:]<=0]))
    

    return py, wavn, theta, rf

def spec_2d(t, z, y1, y2 = None, pad = True):
    """
    Perform wavenumber-frequency spectrum: depth must be 2nd index and time 1st.
    """
    
    t = np.asarray(t)
    z = np.asarray(z)
    if y2 == None:
        y = np.asarray(y1)
    else:
        y = np.asarray(y1 + 1j*y2)
    
    if pad == True:
        y = zero_pad(y, ax = 0)
        y = zero_pad(y, ax = 1)
    
    Nt, Nz = map(len, (y, y.T))
    dt = np.diff(t).mean()
    dz = np.diff(z).mean()
    fy = np.fft.fftshift(np.fft.fft2(y))
    freq = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
    df = 1./(Nt * dt)
    wavn = np.fft.fftshift(np.fft.fftfreq(Nz, dz))
    dm = 1./(Nz * dz)
    py = (dt/Nt)*(dz/Nz)*np.abs(fy)**2
    return py, freq, wavn
    
def spec_2d_filter(t, z, y, cutoffs, quad = None, ftype = ['low', 'low'], tapers = [3, 1], taperpower = 1.5):
    """
    Perform wavenumber-frequency spectrum: depth must be 2nd index.
    cutoff must be a tuple which 1st dim contain 2 or 4 freqs for bandpass or 1 freq 
    for a low pass (default) or high pass, the 2nd dim similarly contain cutoff wavenumbers
    whose order matches the cutoff freqs.
    tapers is a list with 2 elements given the integer width freq/wavenumbers of tapering
    quad is either odds (I and III, -f with -wavn and +f with +wavn), evens or none.
    NOTE: FFT filters are best for broadband sginals of interest.
    """
    assert len(cutoffs) == 2, "cutoffs must be a tuple with size 2 (can be empty)."
    assert quad == 'odds' or quad == 'evens' or quad == None, "bad choice of quadrant."
    
    t = np.asarray(t)
    z = np.asarray(z)
    y = np.asarray(y)
    
    Nt, Nz = map(len, (t, z))
#    dt = (t[1] - t[0])
    dt = np.diff(t).mean()
#    dz = z[1] - z[0]
    dz = np.diff(z).mean()
    fy = np.fft.fftshift(np.fft.fft2(y))
    freq = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
    df = 1./(Nt * dt)
    wavn = np.fft.fftshift(np.fft.fftfreq(Nz, dz))
    dm = 1./(Nz * dz)
    # filter: cosine **1.5 tapered edge window
    #w = np.ones_like(fy.real)
    wt = np.zeros((Nt, 1))
    wz = np.zeros((Nz, 1))
    if len(cutoffs[0]) == 4:
        wt[np.logical_and(freq>=cutoffs[0][0], freq<=cutoffs[0][1])] = 1.
        wt[np.logical_and(freq>=cutoffs[0][2], freq<=cutoffs[0][3])] = 1.
        for n in range(0, len(cutoffs[0])):
            io = find_nearest(freq, cutoffs[0][n] - tapers[0] * df)[1]
            ie = find_nearest(freq, cutoffs[0][n] + tapers[0] * df)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            if io<=ie and n % 2 == 0:
                wt[io:ie+1] = np.flipud(hk)
            elif io<=ie and n % 2 == 1:
                wt[io:ie+1] = hk
            elif io>ie and n % 2 == 0:
                wt[ie:io+1] = np.flipud(hk)
            elif io>ie and n % 2 ==1:
                wt[ie:io+1] = hk
            else:
                raise ValueError, "Something wrong with frequency cutoff choices."
            
        
    elif len(cutoffs[0]) == 2:
        wt[np.logical_and(freq>=cutoffs[0][0], freq<=cutoffs[0][1])] = 1.
        for n in range(0, len(cutoffs[0])):
            io = find_nearest(freq, cutoffs[0][n] - tapers[0] * df)[1]
            ie = find_nearest(freq, cutoffs[0][n] + tapers[0] * df)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            if io<=ie and n % 2 == 0:
                wt[io:ie+1] = np.flipud(hk)
            elif io<=ie and n % 2 == 1:
                wt[io:ie+1] = hk
            elif io>ie and n % 2 == 0:
                wt[ie:io+1] = np.flipud(hk)
            elif io>ie and n % 2 ==1:
                wt[ie:io+1] = hk
            else:
                raise ValueError, "Something wrong with frequency cutoff choices."
            
        
    elif len(cutoffs[0]) == 1:
        if ftype[0] == 'low':
            wt[np.abs(freq)<=np.abs(cutoffs[0][0])] = 1.
            io = find_nearest(freq, -np.abs(cutoffs[0][0]) - tapers[0] * df)[1]
            ie = find_nearest(freq, -np.abs(cutoffs[0][0]) + tapers[0] * df)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            wt[io:ie+1] = np.flipud(hk)
            io = find_nearest(freq, np.abs(cutoffs[0][0]) - tapers[0] * df)[1]
            ie = find_nearest(freq, np.abs(cutoffs[0][0]) + tapers[0] * df)[1]
            wt[io:ie+1] = hk
        elif ftype[0] == 'high':
            wt[np.abs(freq)>=np.abs(cutoffs[0][0])] = 1.
            io = find_nearest(freq, -np.abs(cutoffs[0][0]) - tapers[0] * df)[1]
            ie = find_nearest(freq, -np.abs(cutoffs[0][0]) + tapers[0] * df)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            wt[io:ie+1] = hk
            io = find_nearest(freq, np.abs(cutoffs[0][0]) - tapers[0] * df)[1]
            ie = find_nearest(freq, np.abs(cutoffs[0][0]) + tapers[0] * df)[1]
            wt[io:ie+1] = np.flipud(hk)
        else:
            raise ValueError, "for one frequency, pick high or low pass ftype."
        
    elif len(cutoffs[0]) == 0:
        wt = np.ones((Nt, 1))
    
    else:
        raise ValueError, "frequency cutoffs must number either 4, 2, 1 or 0."
    
    #    wavenumbers:
    if len(cutoffs[1]) == 4:
        wz[np.logical_and(wavn>=cutoffs[1][0], wavn<=cutoffs[1][1])] = 1.
        wz[np.logical_and(wavn>=cutoffs[1][2], wavn<=cutoffs[1][3])] = 1.
        for n in range(0, len(cutoffs[1])):
            io = find_nearest(wavn, cutoffs[1][n] - tapers[1] * dm)[1]
            ie = find_nearest(wavn, cutoffs[1][n] + tapers[1] * dm)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            if io<=ie and n % 2 == 0:
                wz[io:ie+1] = np.flipud(hk)
            elif io<=ie and n % 2 == 1:
                wz[io:ie+1] = hk
            elif io>ie and n % 2 == 0:
                wz[ie:io+1] = np.flipud(hk)
            elif io>ie and n % 2 ==1:
                wz[ie:io+1] = hk
            else:
                raise ValueError, "Something wrong with wavenumber cutoff choices."
            
        
    elif len(cutoffs[1]) == 2:
        wz[np.logical_and(wavn>=cutoffs[1][0], waven<=cutoffs[1][1])] = 1.
        for n in range(0, len(cutoffs[1])):
            io = find_nearest(wavn, cutoffs[1][n] - tapers[1] * dm)[1]
            ie = find_nearest(wavn, cutoffs[1][n] + tapers[1] * dm)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            if io<=ie and n % 2 == 0:
                wz[io:ie+1] = np.flipud(hk)
            elif io<=ie and n % 2 == 1:
                wz[io:ie+1] = hk
            elif io>ie and n % 2 == 0:
                wz[ie:io+1] = np.flipud(hk)
            elif io>ie and n % 2 ==1:
                wz[ie:io+1] = hk
            else:
                raise ValueError, "Something wrong with wavenumber cutoff choices."
            
        
    elif len(cutoffs[1]) == 1:
        if ftype[1] == 'low':
            wz[np.abs(wavn)<=np.abs(cutoffs[1][0])] = 1.
            io = find_nearest(wavn, -np.abs(cutoffs[1][0]) - tapers[1] * dm)[1]
            ie = find_nearest(wavn, -np.abs(cutoffs[1][0]) + tapers[1] * dm)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            wz[io:ie+1] = np.flipud(hk)
            io = find_nearest(wavn, np.abs(cutoffs[1][0]) - tapers[1] * dm)[1]
            ie = find_nearest(wavn, np.abs(cutoffs[1][0]) + tapers[1] * dm)[1]
            wz[io:ie+1] = hk
        elif ftype[1] == 'high':
            wz[np.abs(wavn)>=np.abs(cutoffs[1][0])] = 1.
            io = find_nearest(wavn, -np.abs(cutoffs[1][0]) - tapers[1] * dm)[1]
            ie = find_nearest(wavn, -np.abs(cutoffs[1][0]) + tapers[1] * dm)[1]
            k = np.linspace(1.e-10, 1., num = np.abs(io - ie) + 1)
            hk = (np.sin(np.pi * k) / (np.pi * k))**taperpower
            hk = hk.reshape(len(hk), 1)
            wz[io:ie+1] = hk
            io = find_nearest(wavn, np.abs(cutoffs[1][0]) - tapers[1] * dm)[1]
            ie = find_nearest(wavn, np.abs(cutoffs[1][0]) + tapers[1] * dm)[1]
            wz[io:ie+1] = np.flipud(hk)
        else:
            raise ValueError, "for one wavenumber, pick either a high or low ftype."
    
    elif len(cutoffs[1]) == 0:
        wz = np.ones((Nz, 1))
    
    else:
        raise ValueError, "wavenumber cutoffs must number either 4, 2, 1 or 0."
    
    wz = wz.reshape(1, Nz)
    w = np.zeros((Nt, Nz))
    if quad == 'odds':
        wn = np.dot(wt[freq<=0], wz[:, wavn<=0])
        wp = np.dot(wt[freq>=0], wz[:, wavn>=0])
        nfidx = np.where(freq<=0)[0]
        pfidx = np.where(freq>=0)[0]
        nwidx = np.where(wavn<=0)[0]
        pwidx = np.where(wavn>=0)[0]
        w[nfidx[0]:nfidx[-1]+1, nwidx[0]:nwidx[-1]+1] = wn
        w[pfidx[0]:pfidx[-1]+1, pwidx[0]:pwidx[-1]+1] = wp
    elif quad == 'evens':
        wn = np.dot(wt[freq<=0], wz[:, wavn>=0])
        wp = np.dot(wt[freq>=0], wz[:, wavn<=0])
        nfidx = np.where(freq<=0)[0]
        pfidx = np.where(freq>=0)[0]
        nwidx = np.where(wavn<=0)[0]
        pwidx = np.where(wavn>=0)[0]
        w[nfidx[0]:nfidx[-1]+1, pwidx[0]:pwidx[-1]+1] = wn
        w[pfidx[0]:pfidx[-1]+1, nwidx[0]:nwidx[-1]+1] = wp
    elif quad == None:
        w = np.dot(wt, wz)
    
    yn = np.fft.ifft2(np.fft.ifftshift(fy * w))
    if y.dtype.kind == 'c':
        return yn, w, wt, wz
    else:
        return yn.real, w, wt, wz


def cross_spec(x, y1, y2, win = None, pad = True, ax = 0):
    """
    Cross spectrum, non-rotary 
    """
    if pad == True:
        y1 = zero_pad(y1, ax = ax)
        y2 = zero_pad(y2, ax = ax)
    
    d = np.diff(x, axis = ax).mean()
    N = y1.shape[ax]
    if not win in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
        raise ValueError, "Window choice is invalid"
    
    if win != None:
        win = eval('windows.' + win + '(N' + ', sym = False)')
        dofw = len(win) / np.sum(win**2)
        win.resize((N,) + tuple(np.int8(np.ones(y1.ndim - 1))))
        if ax != 0 and ax != -1:
            win = np.rollaxis(win, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            win = np.rollaxis(win, 0, start = y1.dim)
        elif ax == 0:
            win = win
        else:
            raise ValueError, "Pick your axis better."
        
        y1 = sci_detrend(y1 * win, axis = ax)
        y2 = sci_detrend(y2 * win, axis = ax)
    else:
        dofw = 1.0
    
    fy1, fy2 = map(np.fft.fft, (y1, y2), (None, None), (ax, ax))
    fy1, fy2 = map(np.fft.fftshift, (np.sqrt(dofw)*fy1, np.sqrt(dofw)*fy2), (ax, ax))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d))

    py1 = (d/N)*np.abs(fy1)**2
    py2 = (d/N)*np.abs(fy2)**2
    py1y2 = (d/N)*(fy1.conj() * fy2) # cross spectrum
    cy1y2 = (d/N)*( fy1.real*fy2.real + fy1.imag*fy2.imag ) # coincident spectrum
    qy1y2 = (d/N)*( fy1.real*fy2.imag - fy2.real*fy1.imag ) # quadrature spectrum

    ay1y2 = np.sqrt( cy1y2**2 + qy1y2**2 ) # cross amplitude
    phase = np.arctan2(-1.*qy1y2, cy1y2)

    return freq, py1y2, cy1y2, qy1y2, ay1y2, py1, py2, phase, dofw 

def rot_cross_spec(x, y1, y2, win = None, pad = True, ax = 0):
    """
    Rotary Cross spectrum: apply to dim = ax by transposing, ffting dim=0
    and then tranposing back
    """
    if pad == True:
        y1 = zero_pad(y1, ax = ax)
        y2 = zero_pad(y2, ax = ax)
    
    d = np.diff(x, axis = ax).mean()
    N = y1.shape[ax]
    if not win in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman', 'triang', None]:
        raise ValueError, "Window choice is invalid"
    
    if win != None:
        win = eval('windows.' + win + '(N)')
        dofw = len(win) / np.sum(win**2)
        win.resize((N,) + tuple(np.int8(np.ones(y1.ndim - 1))))
        if ax != 0 and ax != -1:
            win = np.rollaxis(win, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            win = np.rollaxis(win, 0, start = y1.dim)
        elif ax == 0:
            win = win
        else:
            raise ValueError, "Pick your axis better."
        
        y1 = y1 * win
        y2 = y2 * win
    else:
        dofw = 1.0
    
    if ax != 0:
        y1 = np.rollaxis(y1, ax, start = 0)
        y2 = np.rollaxis(y2, ax, start = 0)
    
    fy1, fy2 = map(np.fft.fft, (y1, y2), (None, None), (0, 0))
    fy1, fy2 = map(np.fft.fftshift, (np.sqrt(dofw)*fy1, np.sqrt(dofw)*fy2))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d))

    ipy1 = (d/N)*np.abs(fy1)**2
    ipy2 = (d/N)*np.abs(fy2)**2
    ipy1y2 = (d/N)*(fy1.conj() * fy2) # inner cross spectrum
    iphase_y1y2 = np.arctan2(-1.*ipy1y2.imag, ipy1y2.real)
    
    opy1 = ipy1 * np.flipud( ipy1 ) # outer autospectrum (not a "real" spec: complex)
    opy2 = ipy2 * np.flipud( ipy2 )
    opy1y2 = ipy1 * np.flipud( ipy2 )
    opy2y1 = ipy2 * np.flipud( ipy1 )
    ophase_y1y2 = np.arctan2( -1.* np.concatenate(np.flipud(opy2y1[freqs>0].imag), opy1y2[freqs>=0].imag, axis = 0), np.concatenate( np.flipud(opy2y1[freqs>0].real), opy1y2[freqs>=0].real, axis = 0) )
    
    if ax != 0:
        ipy1y2 = np.rollaxis(ipy1y2, 0, start = ax)
        iphase_y1y2 = np.rollaxis(iphase_y1y2, 0, start = ax)
        opy1y2 = np.rollaxis(opy1y2, 0, start = ax)
        ophase_y1y2 = np.rollaxis(ophase_y1y2, 0, start = ax)
    
    return freqs, ipy1y2, ipy1, ipy2, iphase_y1y2, opy1y2, opy2y1, opy1, opy2, ophase_y1y2, dofw 

def pwelch(x, y1, y2 = None, Nens = 2, Nband = None, noverlap = True, win = None, pad = True, ax = -1):
    """
    Given a timeseries computes a periodgram with the Welch method and or Band averaging,
    outputs average PSD and statistics in a tuple. Overlap is desired if windowing is to be used.
    """
    return psd, Stats

def lperiodgram(psd, dofw = 1, alpha = 0.05, Nens = 2, Nband = 1, smoo = True, ax = -1):
    """
    Computes a smothed or binned late periodgram with the no-overlap Welch method and/or Band averaging
    for a given array of PSD, and outputs an average PSD along axis ax and its statistics in a tuple.
    """
    if smoo == True:
        #N = np.floor(psd.shape[ax]/int(Nens))
        N = np.floor(int(Nens))
        win = windows.boxcar(N)
        win = win/win.sum()
        win.resize((N,) + tuple(np.int8(np.ones(psd.ndim - 1))))
        if ax != 0 and ax != -1:
            win = np.rollaxis(win, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            win = np.rollaxis(win, 0, start = psd.ndim)
        elif ax == 0:
            win = win
        else:
            raise ValueError, "Pick your axis better."

        mpsd = sci_fftconvolve(psd, win, mode = 'same')
    else:
        mpsd = binav(psd, bins = Nens, ax = ax)
    
    if Nband > 1:
        if Nband % 2 != 1:
            Nband = Nband + 1
        
        wbd = windows.boxcar(Nband)
        wbd = wbd / wbd.sum()
        wbd.resize((Nband,) + tuple(np.int8(np.ones(mpsd.ndim - 1))))
        if ax != 0 and ax != -1:
            wbd = np.rollaxis(wbd, 0, start = ax + 1)
        elif ax != 0 and ax == -1:
            wbd = np.rollaxis(wbd, 0, start = mpsd.ndim)
        elif ax == 0:
            wdb = wbd
        else:
            raise ValueError, "Pick your axis better."
        
        mpsd = sci_fftconvolve(mpsd, wbd, mode = 'same')
    
    dof = 2*Nens*Nband*dofw # for non-overlaping segments
    psd_hi = (dof * mpsd) / (stats.chi2.ppf(.5 * alpha, dof))
    psd_lo = (dof * mpsd) / (stats.chi2.ppf(1-(alpha/2), dof))
    loci = np.log10(dof / stats.chi2.ppf(1-(alpha/2), dof))
    hici = np.log10(dof / stats.chi2.ppf(.5 * alpha, dof))
    mpsd = mpsd
    Stats = tuple([psd_lo, psd_hi, loci, hici, dof])
    return mpsd, Stats

def coherence_spec(x, y1, y2, Nens = 2, Nband = 1, win = 'blackman', pad = True, alpha = .05, ax =0):
    """
    coherence and phase based on Welch periodgram and/or frequency band averaging
    of cross spectra. Periodgram averaging takes place on last dim. Needs reforming.
    """
    if y1.dtype.kind == 'c' and y2.dtype.kind == 'c':
        c_spec = rot_cross_spec(x, y1, y2, win = win, pad = pad, ax = ax)
        # inner coherence squared
        ipy1y2, istats = lperiodgram(c_spec[1], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        ipy1, _ = lperiodgram(c_spec[2], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        ipy2, _ = lperiodgram(c_spec[3], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        icoh = np.abs(ipy1y2)**2 / ( ipy1 * ipy2 )
        iphase = np.arctan2(-1.*iy1y2.imag, iy1y2.real)
        icoh_lo = 1.0 - alpha**(2./(istats[-1]-2))
        iphase_hi = iphase + 2.0*np.sqrt((1.0/istats[-1])*((1.0/icoh)-1.0))
        iphase_lo = iphase - 2.0*np.sqrt((1.0/istats[-1])*((1.0/icoh)-1.0))
        # outer
        opy1y2, ostats = lperiodgram(c_spec[5], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        opy2y1, _ = lperiodgram(c_spec[6], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        opy1, _ = lperiodgram(c_spec[7], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        opy2, _ = lperiodgram(c_spec[8], c_spec[-1], alpha = alpha, Nens = Nens, Nband = Nband)
        freqs = c_spec[0]
        # roll axis again if ax not = 0
        if ax != 0:
            opy1y2 = np.rollaxis(opy1y2, ax, start = 0)
            opy2y1 = np.rollaxis(opy2y1, ax, start = 0)
            ipy1 = np.rollaxis(ipy1, ax, start = 0)
            ipy2 = np.rollaxis(ipy2, ax, start = 0)
        
        ocoh = np.concatenate( opy1y2[freqs<=0].imag**2 + opy2y1[freqs>=0].real**2 , opy2y1[freqs<=0].imag**2 + opy1y2[freqs>=0].real**2, axis = 0 ) / ( np.flipud(ipy1) * ipy2 )
        ophase = np.arctan2( -1.* np.concatenate(np.flipud(opy2y1[freqs>0].imag), opy1y2[freqs>=0].imag, axis = 0), np.concatenate( np.flipud(opy2y1[freqs>0].real), opy1y2[freqs>=0].real, axis = 0) )
        
        if ax != 0:
            opy1y2 = np.rollaxis(opy1y2, 0, start = ax)
            opy2y1 = np.rollaxis(opy2y1, 0, start = ax)
            ipy1 = np.rollaxis(ipy1, 0, start = ax)
            ipy2 = np.rollaxis(ipy2, 0, start = ax)
            ocoh = np.rollaxis(ocoh, 0, start = ax)
            ophase = np.rollaxis(ophase, 0, start = ax)
        
        ophase_hi = ophase + 2.0*np.sqrt((1.0/ostats[-1])*((1.0/icoh)-1.0))
        ophase_lo = ophase - 2.0*np.sqrt((1.0/ostats[-1])*((1.0/icoh)-1.0))
        return freqs, icoh, iphase, ocoh, ophase, icoh_lo, iphase_lo, iphase_hi, ophase_lo, ophase_hi 
    else:
        c_spec = cross_spec(x, y1, y2, win = win, pad = pad, ax = ax)
        py1y2, stats12 = lperiodgram(c_spec[1], c_spec[-1], Nens = Nens, Nband = Nband)
        py1, stats1 = lperiodgram(c_spec[5], c_spec[-1], Nens = Nens, Nband = Nband)
        py2, stats2 = lperiodgram(c_spec[6], c_spec[-1], Nens = Nens, Nband = Nband)
        coh = np.abs(py1y2)**2 / ( py1 * py2 )
        phase = np.arctan2(-1.*py1y2.imag, py1y2.real)
        coh_lo = 1.0 - alpha**(2./(stats12[-1]-2))
        phase_hi = phase + 2.0*np.sqrt((1.0/stats12[-1])*((1.0/coh)-1.0))
        phase_lo = phase - 2.0*np.sqrt((1.0/stats12[-1])*((1.0/coh)-1.0))
        freq = c_spec[0]
        return freq, coh, phase, coh_lo, phase_lo, phase_hi, stats12, py1y2, py1, stats1, py2, stats2
    

def demod(x, y, fp, nT, win = 'triang'):
    """
    complex demods y about fp (pos) with a given low pass cutoff. works on 1st dim only.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - x[0]
#    dt = x[1] - x[0]
    dt = np.diff(x).mean(axis = 0)
    nyq = 0.5 * (1. / dt)
    N = np.round(nT*round(1/np.abs(fp))/dt)-1
    if N % 2 == 0:
        N = N-1
    win = eval('windows.' + win + '(N)')
    win = win/win.sum()
    if x.ndim != y.ndim:
        win = win[:, np.newaxis]
#    if x.ndim < y.ndim:
        x = x.repeat(y.shape[1])
        x = x.reshape(y.shape)
    
    if y.dtype.kind != 'c':
        yc = y * np.exp(-1j * 2.* np.pi * fp * x)
        yn = sci_fftconvolve(yc, win, mode = 'same')
        amp = 2.0*np.sqrt(yn.real**2 + yn.imag**2)
        phase = np.arctan2(yn.imag, yn.real)
        recon = amp * np.cos(2. * np.pi * fp * x + phase)
        return amp, phase, recon
    else:
        ycp = y * np.exp(-1j * 2.* np.pi * np.abs(fp) * x)
        ycn = y * np.exp(1j * 2.* np.pi * np.abs(fp) * x)
        yp = sci_fftconvolve(ycp, win, mode = 'same')
        yn = sci_fftconvolve(ycn, win, mode = 'same')
#        ampp = 2.0*np.sqrt(yp.real**2 + yp.imag**2)
#        ampn = 2.0*np.sqrt(yn.real**2 + yn.imag**2)
        ampp = np.sqrt(yp.real**2 + yp.imag**2)
        ampn = np.sqrt(yn.real**2 + yn.imag**2)
        phasep = np.arctan2(yp.imag, yp.real)
        phasen = np.arctan2(yn.imag, yn.real)
        reconp = ampp * np.exp(1j * (2. * np.pi * fp * x + phasep))
        reconn = ampn * np.exp(-1j * (2. * np.pi * fp * x + phasen))
        return ampp, ampn, phasep, phasen, reconp, reconn
    


def despike(y, thresh, winsize, ax = 0):
    """
    despiking for masked arrays, removes spikes by masking it.
    removes values above thresh * std of moving window with size winsize
    """
    y = np.asanyarray(y)
    N = winsize
    win = np.ones((N,))/N
    mbar = filters.convolve1d(y, win, axis =  ax)
    devs = np.abs(y - mbar)
    mstd = np.sqrt( filters.convolve1d(devs**2, win, axis = ax) )
    yn = np.ma.masked_where(np.abs(y)>=thresh*mstd, y)   
    return yn

def median_despike(y, siz, tol):
    """
    De-spikes a time-series by median. Removes spikes by masking it. Operates only on 2nd dim.
    """
    assert siz % 2 == 1, "Median filter length must be odd."
    y = np.asanyarray(y)
    k2 = (siz - 1) // 2
    io = 0
    ie = k2+1
    med = np.zeros_like(np.asarray(y))
    med[:, io] = np.median(y[:, io:ie], axis = 1) 
    for n in range(1, ie):
        io = io + 1
        ie = ie + 1
        med[:, io] = np.median(y[:, (io-n):ie], axis = 1)
    
    for n in range(1, y.shape[1]-siz + 1):
        io = io + 1
        ie = ie + 1
        med[:, io] = np.median(y[:, (io-k2):ie], axis = 1)
    
    for n in range(1, k2+1):
        io = io + 1
        ie = ie + 1
        med[:, io] = np.median(y[:, (io-k2):(ie-n)], axis = 1)
    
    yn = np.ma.masked_where( np.abs(y - med) >= tol, y )
    return yn

def fastmed_despike(y, siz, tol, ax = -1):
    """
    De-spikes a time-series by median using the amazing Eric's amazing pycurrent tools.
    Removes spikes by masking it.
    """
    assert siz % 2 == 1, "Median filter length must be odd."
    y = np.asanyarray(y)
    x = Runstats(y, siz, axis = ax)
    yn = np.ma.masked_where( np.abs(y - x.median) >= tol, y )
    return yn

def smooth1d(y, winsize, ax = 0, wintype = 'blackman'):
    """
    smooths NON-masked arrays with blackman or boxcar window.
    """
    if not wintype in ['boxcar', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'boxcar', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    y = np.asanyarray(y)
    N = winsize
    win = eval('windows.' + wintype + '(N)')
    yn = filters.convolve1d(y, win/win.sum(), axis =  ax)
    return yn

def binav(y, bins = 1, ax = -1):
    """ bin averages y with bin size bins along axis = -1, if ax not -1 transposes"""
    y = np.asanyarray(y)
    if ax != -1:
        y = np.rollaxis(y, ax, start = y.ndim)
    
    if y.shape[-1] % int(bins) == 1.0 and y.ndim > 1:
        y = y.swapaxes(-1, 0)
        y = y[0:-1]
        y = y.swapaxes(0, -1)
    elif y.shape[-1] % int(bins) > 1.0 and y.ndim > 1:
        b = y.shape[-1] % int(bins)
        y = y.swapaxes(-1, 0)
        y = y[b/2:-b/2]
        y = y.swapaxes(0, -1)
    elif y.shape[-1] % int(bins) > 1.0 and y.ndim == 1:
        b = y.shape[-1] % int(bins)
        y = y[b/2:-b/2]
    elif y.shape[-1] % int(bins) == 1.0 and y.ndim == 1:
        y = y[0:-1]
    a = y.shape[-1] / int(bins)
    newshape = (y.shape[0:-1] + (a,) + (bins,))
    yn = y.reshape(newshape).mean(axis = -1).squeeze()
    if ax != -1:
        yn = np.rollaxis(yn, -1, start = ax)
    
    return yn

def butter_filter(x, y, cutoffs, Btype = 'low', ax = 0, bl = 3, GP = 1, GS = 30, padtype=None, padlen = None):
    """
    low, high or band pass butterworth filter: cutoffs is the low and high frq edges
    of pass band. 
    """
    x = x - x[0]
    if x.ndim == 1:
        dt = np.diff(x).mean()
        df = 1./(x[-1]-x[0])
    else:
        dt = np.diff(x, axis=0).mean()
        df = (1./(x[-1]-x[0]))[0]
    
    nyq = 0.5 * (1. / dt)
    bl = int(bl)
    
    if Btype == 'band':
        order, wc = buttord(cutoffs/nyq, [cutoffs[0]-bl*df, cutoffs[1]+bl*df]/nyq, gpass = GP, gstop = GS)
    elif Btype == 'low':
        order, wc = buttord(cutoffs/nyq, (cutoffs+bl*df)/nyq, gpass = GP, gstop = GS)
    elif Btype == 'high':
        order, wc = buttord(cutoffs/nyq, (cutoffs-bl*df)/nyq, gpass = GP, gstop = GS)
    else:
        raise ValueError, "Btype is low, high or band only."
    
    b, a = butter(order, wc, btype = Btype)
    yn = filtfilt(b, a, y, axis = ax, padtype = padtype, padlen = padlen)
    
    return yn

def butter_demod(x, y, fp, cutoff, bl = 3, GP = 1, GS = 10):
    """
    complex demods y about fp with a given low pass cutoff. works on 1st dim only.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - x[0]
    dt = x[1] - x[0]
    nyq = 0.5 * (1. / dt)
    if x.ndim < y.ndim:
        x = x.repeat(y.shape[1])
        x = x.reshape(y.shape)
    
    if y.dtype.kind != 'c':
        yc = y * np.exp(-1j * 2.* np.pi * fp * x)
        yn = butter_filter(x, yc, cutoff, Btype = 'low', bl = bl, GP = GP, GS = GS)
        amp = 2.0*np.sqrt(yn.real**2 + yn.imag**2)
        phase = np.arctan2(yn.imag, yn.real)
        recon = amp * np.cos(2. * np.pi * fp * x + phase)
        return amp, phase, recon
    else:
        ycp = y * np.exp(-1j * 2.* np.pi * fp * x)
        ycn = y * np.exp(1j * 2.* np.pi * fp * x)
        yp = butter_filter(x, ycp, cutoff, Btype = 'low', bl = bl, GP = GP, GS = GS)
        yn = butter_filter(x, ycn, cutoff, Btype = 'low', bl = bl, GP = GP, GS = GS)
        ampp = np.sqrt(yp.real**2 + yp.imag**2)
        ampn = np.sqrt(yn.real**2 + yn.imag**2)
        phasep = np.arctan2(yp.imag, yp.real)
        phasen = np.arctan2(yn.imag, yn.real)
        reconp = ampp * np.cos(2. * np.pi * fp * x + phasep)
        reconn = ampn * np.cos(2. * np.pi * fp * x + phasen)
        return ampp, ampn, phasep, phasen, reconp, reconn
    


def do_wavelet(x, y, fr, mom = 'Morlet', f0 = 6., dj = .125, s0 = -1, J = -1,
               slevel = .95, plts = True, rectify=False):
    """
    Uses kPyWavelets to perform wavelet analysis of real timeseries y (a 1d array).
    """
    std = y.std()
    var = std ** 2
    y = (y - y.mean()) / std
    N = len(y)
    dt = np.diff(x).mean()

    alpha, _, _ = wavelet.ar1(y)
    
    # the wavelet transform
    mother = eval('wavelet.' + mom + '(' + str(f0) + ')')
    wave, scales, freqs, coi, FFT, FFTfreqs = wavelet.cwt(y, dt, dj, s0, J,\
                                                              mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) # inverse wavelet transform

    power = (np.abs(wave)) ** 2             # Normalized wavelet power spectrum
    fft_power = var * np.abs(FFT) ** 2     # FFT power spectrum
    period = 1. / freqs
    # this rectification may be bullshit, plus if used power must be re-scaled again further down
    if rectify:
        power = power / np.ones([1, N]) * scales[:, None]
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,\
                                             significance_level=slevel, wavelet=mother)
    sig = np.ones([1, N]) * signif[:, None]
    sig = power / sig                # Where ratio > 1, power is significant

    glbl_power = var * power.mean(axis = 1)
    dof = N - scales                     # Correction for padding at edges
    glbl_sig, tmp = wavelet.significance(var, dt, scales, 1, alpha,\
                                             significance_level=slevel, dof=dof, wavelet=mother)
    
    # Scale average between avg1 and avg2 periods and significance level
    sel = find((freqs >= fr[0]) & (freqs <= fr[1]))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    scale_avg = power / scale_avg
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis = 0)
    scale_avg_sig, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                              significance_level=slevel, dof=[scales[sel[0]],
                                                                              scales[sel[-1]]], wavelet=mother)
    
    if plts == True:
        figprops = dict(figsize=(11, 8), dpi=72)
        figure(**figprops)
        
        ax = axes([0.1, 0.75, 0.65, 0.2])
        ax.plot(x, y, 'k', lw = 1.5)
        ax.plot(x, iwave, '-', lw = 1, color=[0.5, 0.5, 0.5])
        ax.set_title('Normalized Anomaly timeseries and its inverse wavelet transform (gray)')
         
        bx = axes([0.1, 0.37, 0.65, 0.28], sharex = ax)
        bx.contourf(x, freqs, power, extend='both')
        bx.contour(x, freqs, sig, [-99, 1], colors='k', lw = 2.)        
        bx.fill(np.concatenate([[x[0]-dt, x[0]-dt], x[0:-1], [x[-1]+dt, x[-1]+dt]]),
                np.concatenate([[1e-9, 1./coi[0]], 1./coi[0:-1], [1./coi[-2], 1e-9]]),
                'k' , alpha='0.3', hatch='x')
        bx.set_ylim([0, 2*fr[1]])
        bx.set_ylabel('Frequency')
        bx.set_title('Wavelet power spectrum (' + mom + ')')
        
        cx = axes([0.77, 0.37, 0.2, 0.28])
        cx.semilogx(fft_power, FFTfreqs, '-', color=[0.7, 0.7, 0.7])
        cx.semilogx(glbl_power, freqs, 'k-', linewidth=1.5)
        cx.plot(glbl_sig, freqs, 'k--')
        cx.set_ylim([0, 2*fr[1]])
        cx.set_title('Global wavelet spec (thick) vs FFT spec')
        cx.set_xlabel('Power')
        cx.set_yticklabels(cx.get_yticklabels(), visible = False)
        
        dx = axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.plot(x, scale_avg, 'k-', linewidth=1.5)
        dx.axhline(scale_avg_sig, color='k', linestyle='--', linewidth=1.)
        dx.set_title('$%f$-$%f$ freq band scale-averaged power' % (fr[0], fr[1]))
        dx.set_ylabel('Average variance')
        draw()
        show()
    
    return freqs, power, sig, coi, iwave, glbl_power, glbl_sig, scale_avg, scale_avg_sig, scales

def do_cross_wavelet(x1, y1, x2, y2, mom = 'Morlet', f0 = 6., dj = .125, slevel = .95,
                     s0 = -1, J = -1, plts = True, fr = [1./6., .5], tj = 36):
    """
    Uses kPyWavelets to perform cross wavelet analysis of timeseries y1 and y2 (both 1d
    arrays.
    """
    std1 = y1.std()
    std2 = y2.std()
    var1 = std1 ** 2
    var2 = std2 ** 2
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()
    N1, N2 = map(len, (y1, y2))
    n = np.min((N1, N2))
    if x1.tolist() == x2.tolist():
        x = x1
    else:
        x = x1 - x2
    
    dt = np.diff(x).mean() # both series must have same sampling rate

    alpha1, _, _ = wavelet.ar1(y1)
    alpha2, _, _ = wavelet.ar1(y2)

    mother = eval('wavelet.' + mom + '(' + str(f0) + ')')
    kwargs = {'dt':dt, 'dj':dj, 's0':s0, 'J':J, 'wavelet':mother}

    cwt1 = wavelet.cwt(y1 / std1, **kwargs)
    sig1 = wavelet.significance(1., dt, cwt1[1], 0, alpha1, 
                                significance_level=slevel, wavelet=mother)
    cwt2 = wavelet.cwt(y2 / std2, **kwargs)
    sig2 = wavelet.significance(1., dt, cwt2[1], 0, alpha2, 
                                significance_level=slevel, wavelet=mother)
    power_y2 = var2 * np.abs(cwt2[0]) ** 2
    sig_y2 = np.ones([1, n]) * sig2[0][:, None]
    sig_y2 = power_y2 / var2 / sig_y2

    # my interest is energy (actual variance in each scale: as in Torrence and Compo (1998) equation 24)
    Cdelta = mother.cdelta
    scale_avg = (cwt2[1] * np.ones((n, 1))).transpose()
    energies_y2 = power_y2 / scale_avg
    energies_y2 = dj * dt / Cdelta * energies_y2
    
    xwt = wavelet.xwt(x1, y1, x2, y2, significance_level=slevel, normalize=True, **kwargs)
    wct = wavelet.wct(x1, y1, x2, y2, significance_level=slevel, normalize=True, **kwargs)
    
    xpower = np.abs(xwt[0])
    freqs = xwt[3]
    sig = np.ones([1, n]) * xwt[4][:, None]
    sig = xpower / sig
#    angle = 0.5 * np.pi - np.angle(xwt[0]) # so phase rotates clockwise with 'north' origin.
    angle = np.angle(xwt[0])
    u, v = np.cos(angle), np.sin(angle)
    
    coh = wct[0]
    csig = np.ones([1, n]) * wct[4][:, None]
    csig = coh / csig
#    cangle = .5 * np.pi - wct[5]
    cangle = wct[5]
    cu, cv = np.cos(cangle), np.sin(cangle)
    
    if plts == True:
        figx = plt.figure()
        ax = figx.add_subplot(2,1,1)
        coi = xwt[2]
        ax.contourf(x, freqs, xpower)
        ax.contour(x, freqs, sig, [-99, 1], colors='k', linewidths=2.)
        ax.fill(np.concatenate([[x[0]-dt, x[0]-dt], x[0:-1], [x[-1]+dt, x[-1]+dt]]),
                np.concatenate([[1e-9, 1./coi[0]], 1./coi[0:-1], [1./coi[-2], 1e-9]]),
                'k' , alpha='0.3', hatch='x')
        ax.quiver(x[::tj], freqs, u[:, ::tj], v[:, ::tj], units='width', angles='uv', 
                  pivot='mid', linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                  headaxislength=5, minshaft=2, minlength=5)
        ax.set_ylim([0, 2*fr[1]])
        ax.set_ylabel('Frequency')
        bx = figx.add_subplot(2,1,2)
        cf = bx.contourf(x, freqs, angle)
        bx.quiver(x[::tj], freqs, u[:, ::tj], v[:, ::tj], units='width', angles='uv', 
                  pivot='mid', linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                  headaxislength=5, minshaft=2, minlength=5)
        cbar = figx.colorbar(cf)
        bx.set_ylim([0, 2*fr[1]])
        bx.set_ylabel('Frequency')
        bx.set_xlabel('Time')
        
        figc = plt.figure()
        cx = figc.add_subplot(2,1,1)
        cff = cx.contourf(x, freqs, coh)
        cx.contour(x, freqs, csig, [-99, 1], colors='k', linewidths=2.)
        cx.fill(np.concatenate([[x[0]-dt, x[0]-dt], x[0:-1], [x[-1]+dt, x[-1]+dt]]),
                np.concatenate([[1e-9, 1./coi[0]], 1./coi[0:-1], [1./coi[-2], 1e-9]]),
                'k' , alpha='0.3', hatch='x')
        cx.quiver(x[::tj], freqs, cu[:, ::tj], cv[:, ::tj], units='width', angles='uv', 
                  pivot='mid', linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                  headaxislength=5, minshaft=2, minlength=5)
        figc.colorbar(cff)
        cx.set_ylim([0, 2*fr[1]])
        cx.set_ylabel('Frequency')
        dx = figc.add_subplot(2,1,2)
        cf2 = dx.contourf(x, freqs, cangle)
        dx.quiver(x[::tj], freqs, cu[:, ::tj], cv[:, ::tj], units='width', angles='uv', 
                  pivot='mid', linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
                  headaxislength=5, minshaft=2, minlength=5)
        cbar2 = figc.colorbar(cf2)
        dx.set_ylim([0, 2*fr[1]])
        dx.set_ylabel('Frequency')
        dx.set_xlabel('Time')
        
    return freqs, xpower, sig, angle, coh, csig, energies_y2, sig_y2, power_y2, xwt, wct, cwt1, cwt2

def regrid(t, z, y, ti = None, zi = None, method = 'linear'):
    """ interpolate y[t, z] to yn[ti, zi] or yn[t, zi] or yn[ti, z]"""
    if ti != None and zi != None:
        """ 2d interpolation """
        yn = griddata(t, z, y, ti, zi, interp = method)
    elif ti != None and zi == None:
        """ time interpolation """
        yn = np.empty((len(ti), len(z)))
        for n in range(0, len(z)):
            yn[:, n] = np.interp(ti, t, y[:, n])
    elif ti == None and zi != None:
        """ depth interpolation """
        yn = np.empty((len(t), len(zi)))
        for n in range(0, len(t)):
            yn[n, :] = np.interp(zi, z, y[n, :])
        
    return yn

def calc_ip_f0(lat):
    """ Calculates inertial period and f0 for given latitude in degrees"""
    omega = 7.2921e-5
    f0 = 2 * omega * np.sin( np.deg2rad(lat) )
    ip =  2*np.pi/np.abs(f0)/3600/24
    return ip, f0

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def plotyy(x1, y1, x2, y2, plt1_str='-b', plt2_str='-r'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1, y1, plt1_str)
# Make the y-axis label and tick labels match the line color.
    for tl in ax1.get_yticklabels():
        tl.set_color(plt1_str[-1])
    
    ax2 = ax1.twinx()
    ax2.plot(x2, y2, plt2_str)
    for tl in ax2.get_yticklabels():
        tl.set_color(plt2_str[-1])
    plt.show()

    return fig, ax1, ax2

def plotxx(x1, y1, x2, y2, plt1_str='-b', plt2_str='-r'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1, y1, plt1_str)
# Make the y-axis label and tick labels match the line color.
    for tl in ax1.get_xticklabels():
        tl.set_color(plt1_str[-1])
    
    ax2 = ax1.twiny()
    ax2.plot(x2, y2, plt2_str)
    for tl in ax2.get_xticklabels():
        tl.set_color(plt2_str[-1])
    plt.show()

    return fig, ax1, ax2

def fir_filter(x, y, cutoff, win = 'blackman', ftype = 'low', ntaps = 1001, ax = 0, mode = 'same'):
    """
    Low, High or Bandpass using a FIR filter with a given cutoff fp using scipy fftconvolve
    """
    d = np.diff(x).mean(axis = ax)
    nyq = 1. / (2*d)

    # ideally you would pick a band and the pass/stop gain/loss and a function would give ntaps, etc
    #    N, beta = kaiserord(ripple_db, width)
    #    taps = firwin(ntaps, cutoff/nyq, window=('kaiser', beta))

    if ftype == 'band' or ftype == 'high':
        f = firwin(ntaps, cutoff/nyq, window = win, pass_zero = False)
    elif ftype == 'low':
        f = firwin(ntaps, cutoff/nyq, window = win)
    else:
        raise ValueError, "Pick filter type as low, high or band."
    delay = 0.5 * (ntaps-1) / nyq
    #yn2 = lfilter(f, 1., y, axis = ax)
    #yn3 = filtfilt(f, [1.], y, axis = ax, padtype = None)
    f.resize((ntaps,) + tuple(np.int8(np.ones(y.ndim - 1))))
    if ax != 0 and ax != -1:
        f = np.rollaxis(f, 0, start = ax + 1)
    elif ax != 0 and ax == -1:
        f = np.rollaxis(f, 0, start = y.ndim)
    elif ax == 0:
        f = f
    else:
        raise ValueError, "Pick your axis better."
    
    yn = sci_fftconvolve(y, f, mode = mode)
    return yn

def smooth2d(y, N, ny = None):
    """
    blurs/smooths the image by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction. From the cookbook.
    """
    gk = gauss_kern(N, sizey = ny)
#    improc = sci_convolve(y, gk, mode='valid')
    improc = sci_convolve(y, gk, mode='same')
    
    return improc
