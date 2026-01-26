from pinballrt.sources import Star
from pinballrt.grids import UniformCartesianGrid
from pinballrt.model import Model
import astropy.units as u
import numpy as np

star = Star()
star.set_blackbody_spectrum()

model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":9, "dx":2.0*u.au})
density = np.ones(model.grid.shape)*1.0e-16 * u.g/u.cm**3

model.add_density(density, "yso.dst")
model.add_star(star)

model.thermal_mc(nphotons=1000000, device="cuda")
