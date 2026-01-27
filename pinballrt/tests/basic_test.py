from pinballrt.sources import Star
from pinballrt.grids import UniformCartesianGrid
from pinballrt.model import Model
from pinballrt.densities.flared_disk import FlaredDisk
import astropy.units as u
import numpy as np

star = Star()
star.set_blackbody_spectrum()

model = Model(grid=UniformCartesianGrid, grid_kwargs={"ncells":100, "dx":2.0*u.au})
# density = np.ones(model.grid.shape)*1.0e-16 * u.g/u.cm**3
disk = FlaredDisk()

#model.add_density(density, "yso.dst")
model.add_component(disk, dust="yso.dst")
model.add_star(star)

print(np.max(model.grid.grid.density.numpy()))

model.thermal_mc(nphotons=10000, device="cuda")
