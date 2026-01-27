import numpy as np
import astropy.units as u
from scipy.integrate import nquad

class FlaredDisk():

    def __init__(self, mass=1e-4*u.Msun, rmin=0.1*u.au, rmax=100*u.au, p=-1, beta=1.25, 
                 r_0=1*u.au, h_0=0.1*u.au):
        self.mass = mass # gas mass
        self.rmin = rmin
        self.rmax = rmax
        self.p = p
        self.beta = beta
        self.r_0 = r_0
        self.h_0 = h_0

        self.gamma = self.p + self.beta

    
    def surface_density(self, r):
        sigma0 = ((2 - self.gamma) * self.mass) / (2 * np.pi * self.rmax**2)
        return sigma0 * (r/self.rmax)**(-self.gamma) * np.exp(-(r/self.rmax)**(2-self.gamma))
    
    def scale_height(self, r):
        return self.h_0 * (r / self.r_0)**self.beta
    
    def density(self, r, z):
        sigma = self.surface_density(r)
        h = self.scale_height(r)
        return (sigma / (np.sqrt(2*np.pi)*h)*np.exp(-0.5*(z/h)**2))
    
    def density_grid(self, grid):
        # get grid walls
        w1 = np.array(grid.grid.w1)
        w2 = np.array(grid.grid.w1)
        w3 = np.array(grid.grid.w3)

        # get grid centers
        x1 = (w1[:-1] + w1[1:]) / 2.
        x2 = (w2[:-1] + w2[1:]) / 2.
        x3 = (w3[:-1] + w3[1:]) / 2.

        # get coordinate system
        coordsys = grid.coordsys

        # transform the grid coordinates if necessary
        if coordsys == "spherical":
            rt, tt, pp = np.meshgrid(x1, x2, x3, indexing='ij')
            rr = rt*np.sin(tt)
            zz = rt*np.cos(tt)
        elif coordsys == "cartesian":
            xx, yy, zz = np.meshgrid(x1, x2, x2, indexing='ij')
            rr = np.sqrt(xx**2 + yy**2)
        elif coordsys == "cylindrical":
            rr, pp, zz = np.meshgrid(x1, x2, x2, indexing='ij')

        # calculate the density grid
        return self.density(rr, zz)
        



