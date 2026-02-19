import numpy as np
import astropy.units as u
from scipy.integrate import nquad
from ..component import Component

class FlaredDisk(Component):

    default_params = {"LogM": {"value":-5, "range":(-6, -2), "fixed":False, "units":u.Msun},
                      "LogR_min": {"value":-1., "range":(-3., 0.), "fixed":False, "units":u.au},
                      "LogR_max": {"value":2., "range":(1., 3.), "fixed":False, "units":u.au},
                      "p": {"value":-1, "range":(0., -2.), "fixed":False, "units":None}, 
                      "beta": {"value":1.25, "range":(1., 2.), "fixed":False, "units":None},
                      "R_0": {"value": 1., "range": (0., 10.), "fixed": True, "units":u.au},
                      "LogH_0": {"value":-1, "range":(-2, 1), "fixed":False, "units":u.au},}
    
    def __init__(self):
        super().__init__(name="flared_disk", params=self.default_params)

    def print_param_names(self):
        names = [param for param in self.default_params]
        print(names)
            

    def surface_density(self, r):
        self.gamma = self.p + self.beta
        sigma0 = ((2 - self.gamma) * self.M) / (2 * np.pi * self.R_max**2)
        sigma = (sigma0 * (r/self.R_max)**(-self.gamma) * np.exp(-(r/self.R_max)**(2-self.gamma))).to(u.g/u.cm**2)
        sigma[r < self.R_min] = 0.0
        return sigma
    
    def scale_height(self, r):
        return self.H_0 * (r / self.R_0)**self.beta
    
    def density(self, r, z):
        sigma = self.surface_density(r)
        h = self.scale_height(r)
        return (sigma / (np.sqrt(2*np.pi)*h)*np.exp(-0.5*(z/h)**2)).to(u.g/u.cm**3)
    
    def density_grid(self, grid):
        # get grid walls
        w1 = grid.grid.w1.numpy()
        w2 = grid.grid.w2.numpy()
        w3 = grid.grid.w3.numpy() 

        # get grid centers
        x1 = (w1[:-1] + w1[1:]) / 2.
        x2 = (w2[:-1] + w2[1:]) / 2.
        x3 = (w3[:-1] + w3[1:]) / 2.

        # get coordinate system
        coordsys = grid.coordsys

        # transform the grid coordinates if necessary
        if coordsys == "spherical":
            rt, tt, pp = np.meshgrid(x1, x2, x3, indexing='ij')
            rr = rt*np.sin(tt) * u.au
            zz = rt*np.cos(tt) * u.au
        elif coordsys == "cartesian":
            xx, yy, zz = np.meshgrid(x1, x2, x2, indexing='ij')
            rr = np.sqrt(xx**2 + yy**2) * u.au
            zz *= u.au
        elif coordsys == "cylindrical":
            rr, pp, zz = np.meshgrid(x1, x2, x2, indexing='ij')
            rr += u.au
            zz *= u.au

        # calculate the density grid
        return self.density(rr, zz)
        



