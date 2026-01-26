import numpy as np
import astropy.units as u
from scipy.integrate import nquad

class FlaredDisk():

    def __init__(self, mass, rmin, rmax, p, beta, r_0, h_0):
        self.mass = mass # gas mass
        self.rmin = rmin
        self.rmax = rmax
        self.p = p
        self.beta = beta
        self.r_0 = r_0
        self.h_0 = h_0

        self.gamma = self.p - self.beta

    
    def surface_density(self, r):
        sigma0 = ((2 - self.gamma) * self.mass) / (2 * np.pi * self.rmax**2)
        return sigma0 * (r/self.rmax)**(-self.gamma) * np.exp(-(r/self.rmax)**(2-self.gamma))
    
    def scale_height(self, r):
        return self.h_0 * (r / self.r_0)**self.beta
    
    def density(self, r, z):
        sigma = self.surface_density(r)
        h = self.scale_height(r)

        return (sigma / (np.sqrt(2*np.pi)*h)*np.exp(-0.5*(z/h)**2))
