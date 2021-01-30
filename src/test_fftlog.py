#!/usr/bin/env python
import hankl
import numpy as np
import matplotlib as mpl
from numpy.lib.twodim_base import _trilu_dispatcher
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import rfft, irfft
from scipy.special import loggamma
import sys


def windowfn(x, dlnxleft=0.46, dlnxright=0.46):
  xmin = min(x)
  xmax = max(x)
  xleft = np.exp(np.log(xmin) + dlnxleft)
  xright = np.exp(np.log(xmax) - dlnxright)
  w = np.zeros_like(x)
  w[(x > xleft) & (x < xright)] = 1

  il = (x < xleft) & (x > xmin)
  ir = (x > xright) & (x < xmax)

  rl = (x[il] - xmin) / (xleft - xmin)
  rr = (xmax - x[ir]) / (xmax - xright)
  w[il] = rl - np.sin(np.pi * 2 * rl) / (2 * np.pi)
  w[ir] = rr - np.sin(np.pi * 2 * rr) / (2 * np.pi)
  return w

def calc_Mellnu(tt, alpha, q=0):
  n = q - 1 - 1j * tt
  intjlttn = 2**(n-1) * np.sqrt(np.pi) * \
             np.exp(loggamma((1+n)/2.0) - loggamma((2-n)/2.0))
  A = alpha**(1j * tt - q)
  return A * intjlttn

def calc_phi(pk, k0, N, L, q):
  k = k0 * np.exp(np.arange(0,N) * 2 * np.pi / L)
  P = pk(k)
  kpk = (k / k0)**(3-q) * P * windowfn(k)
  phi = np.conj(rfft(kpk)) / L
  phi *= windowfn(k)[len(k) - len(phi):]
  return phi

def xicalc(pk, N=1000, kmin=1e-4, kmax=1e-4, r0=1e-4):
  '''Arguments:
  pk: callable
  N: number of grids for FFT
  kmin, kmax: k range
  r0: minimum r value (~1/kmax)
  '''
  qnu = 1.95
  N2 = int(N / 2) + 1
  k0 = kmin
  G = np.log(kmax / kmin)
  alpha = k0 * r0
  L = 2 * np.pi * N / G

  tt = np.arange(0, N2) * 2 * np.pi / G
  rr = r0 * np.exp(np.arange(0, N) * (G / N))
  prefac = k0**3 / (np.pi * G) * (rr / r0)**(-qnu)

  Mellnu = calc_Mellnu(tt, alpha, qnu)
  phi = calc_phi(pk, k0, N, L, qnu)
  print(len(rr), len(Mellnu))

  xi = prefac * irfft(phi * Mellnu, N) * N
  return rr, xi



def xi_model_FFTlog(k, P, num_lnk_bin=2048, kmin=5e-3, kmax = 600):
  '''Compute the template correlation function.
  Arguments:
    k, Plin, Pnw: arrays for the linear power spectra;
    Prt: the ratio between the void and linear non-wiggle power spectra;
    nbin: s bins;
    Snl: the BAO damping factor;
    c: the parameter for modeling void non-wiggle power spectrum;
    k2, lnk, eka2, j0: pre-computed values for the model.
  Return: xi_model.'''
  Pm = P
  Pint = interp1d(np.log(k), np.log(Pm), kind='cubic')
  Pkfn = lambda k : np.exp(Pint(np.log(k)))
  s0, xi0 = xicalc(Pkfn, num_lnk_bin, kmin, kmax, 1e-4)
  return s0, xi0/s0**2






linear_pk_fn = 'data/Albert_Plin.dat'
template_pk_fn = 'data/PKvoid_template_case1.pspec'
xi_obs_fn = 'test/mocks_void/TwoPCF_CATALPTCICz0.466G960S1005638091.VOID.R-15.6-50.dat'

s_obs, xi_obs = np.loadtxt(xi_obs_fn, usecols=(0, 1), unpack=True)

kmin=2.5e-3
kmax = 600
lnk = np.log(np.logspace(np.log(kmin), np.log(kmax), 2048, base=np.e))
lnk = np.linspace(np.log(kmin), np.log(kmax), 2048)
print(lnk)


for f in [linear_pk_fn, template_pk_fn]:
    k0, pk = np.loadtxt(f, usecols=(0,1), unpack=True)
    print(f, k0[0])
    
    pkinterp = interp1d(np.log(k0), np.log(pk), kind='linear', bounds_error=False, fill_value='extrapolate')
    k = np.exp(lnk)
    P = np.exp(pkinterp(lnk))
    
    r, xi = hankl.P2xi(k, P, 0, n=0, lowring=_trilu_dispatcher)
    #s, xic = xi_model_FFTlog(k, P)
    print(r)
    plt.plot(r, r**2*xi)
    plt.xlim(0, 200)
    plt.ylim(-100, 200)
    #plt.loglog(k0, pk, label = "orig", marker = '+')
    #plt.loglog(k, P, label = "interp", marker='o', markersize=1)
    plt.legend()

plt.plot(s_obs, s_obs**2*xi_obs)
plt.savefig("test/test_fftlog.png")

