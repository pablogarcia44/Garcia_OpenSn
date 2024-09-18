# Garcia_OpenSn


## Pincell study

This directory contains various analyses on a pin cell without a gap.

### Legendre Order

[Legendre Order](./pincell_study/legendre)

OpenMC input to generate order 7 mgxs, 
Spydermesh input that generate mesh used for all pincell study (mesh_1fuel_fine),
OpenSn input.

### OpenMC_MGXS

[OpenMC_MGXS](./pincell_study/openmc_mgxs)

Contain input file for OpenMC running in MGXS mode for XMAS-172, SHEM-361 and SHEM-361 with different fuel zones.

### Mesh

[spydermsh vs gmsh](./pincell_study/mesh)

Contain all spydermesh and gmsh input to generate the mesh that are used for the mesh study,
need add opensn input when cluster avail


## Assembly study

This directory contains various analyses on a fuel assembly.

### Family

[Family](./assembly_study/family)

Contain OpenMC input to generate mgxs for 1 and 6 families,
spydermesh input to generate assembly mesh with gap water,
Opensn input to add when cluster avail

### Power

[power](./assembly_study/power)

Contain OpenMC input to generate power map,
Opensn input to add when cluster avail

## Benchmark

### MGXS

[mgxs](./benchmark/mgxs)

Contain OpenMC input to generate 1 family mgxs for all 2D CASL VERA benchmark

### Power

[power](./benchmark/power)

Contain OpenMC input to generate power map for all 2D CASL VERA benchmark

### Spidermesh

[spideresh](./benchmark/spydermesh)

Contain a spydermesh_driver input to generate all mesh for 2D CASL VERA benchmark

## Utilities

[utilities](./utilities)

order_alphab.ipynb : Sort a list in alphabetical order, it is useful for creating an OpenSn input file.

plot_h5.ipynb : Allows plotting scattering matrices from a .h5 file.

plot_result.ipynb : All the results obtained.
































