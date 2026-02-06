import numpy as np
import astropy.units as u
from scipy.integrate import nquad

class FlaredDisk():

    def __init__(self, params=None):

        self._default_params = {"LogMass": {"value":-4, "range":(-6, -2), "fixed":False, "units":u.Msun},
                "LogRmin": {"value":-1., "range":(-3., 0.), "fixed":False, "units":u.au},
                "LogRmax": {"value":2., "range":(1., 3.), "fixed":False, "units":u.au},
                "p": {"value":-1, "range":(0., -2.), "fixed":False, "units":None},
                "beta": {"value":1.25, "range":(1., 2.), "fixed":False, "units":None},
                "R_0": {"value": 1., "range": (0., 10.), "fixed": True, "units":u.au},
                "LogH_0": {"value":-1, "range":(-2, 1), "fixed":False, "units":u.au},
                }
        
        self.params = self._default_params
        if params:
            self.set_params(params)

        self.mass = (10**self.params["LogMass"]["value"]) / 100. * self.params["LogMass"]["units"]
        self.rmin = (10**self.params["LogRmin"]["value"]) * self.params["LogRmin"]["units"]
        self.rmax = (10**self.params["LogRmax"]["value"]) * self.params["LogRmax"]["units"]
        self.p = self.params["p"]["value"]
        self.beta = self.params["beta"]["value"]
        self.r0 = self.params["R_0"]["value"] * self.params["R_0"]["units"]
        self.h0 = (10**self.params["LogH_0"]["value"]) * self.params["LogH_0"]["units"]

        self.gamma = self.p + self.beta
        self.name = "Flared Disk"

    def set_params(self, params=None, value=None, range=None, fixed=None, units=None):
        if type(params) is dict:
            for param in params:
                self.params[param] = params[param]
        elif type(params) is str:
            if value:
                self.params[params]["value"] = value
            if range:
                self.params[params]["range"] = range
            if fixed is not None:
                self.params[params]["fixed"] = fixed
            if units:
                self.params[params]["units"] = units


    def print_param_names(self):
        names = [param for param in self._default_params]
        print(names)
            

    def surface_density(self, r):
        sigma0 = ((2 - self.gamma) * self.mass) / (2 * np.pi * self.rmax**2)
        sigma = (sigma0 * (r/self.rmax)**(-self.gamma) * np.exp(-(r/self.rmax)**(2-self.gamma))).to(u.g/u.cm**2)
        sigma[r < self.rmin] = 0.0
        return sigma
    
    def scale_height(self, r):
        return self.h0 * (r / self.r0)**self.beta
    
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
        



