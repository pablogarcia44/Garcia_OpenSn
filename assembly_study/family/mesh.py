C=[(0,0)]
E=[(1,0),(0,1),(2,0),(0,2),(3,0),(0,3),(4,0),(0,4),(5,0),(0,5),(6,0),(0,6),(7,0),(0,7),(8,0),(0,8)]
N=[(1,1),(2,1),(1,2),(3,1),(1,3)]
D=[(1,4),(4,1),(1,5),(6,1),(1,6),(7,1),(1,7),(2,2),(7,3),(3,7),(4,4),(6,4),(4,6),(7,4),(4,7),(6,6),(7,6),(6,7),(7,7),(6,3),(3,6)]
F=[(5,1),(1,5),(8,1),(1,8),(3,2),(2,3),(2,6),(6,2),(2,7),(7,2),(5,3),(3,5),(8,3),(3,8),(5,4),(4,5),(8,4),(4,8),(6,5),(5,6),(7,5),(5,7),(8,6),(6,8),(8,7),(7,8)]
FD=[(4,2),(2,4),(4,3),(3,4)]
GT=[(5,2),(8,2),(3,3),(2,5),(5,5),(8,5),(2,8),(5,8)]
IT=[(8,8)]

E_left=[]
E_full=[]
for X in E:
    E_left.append((X[0],X[1]))
    E_left.append((X[0],16-X[1]))
for X in E_left:
    E_full.append((X[0],X[1]))
    E_full.append((16-X[0],X[1]))
E_full = list(set(E_full))

C_left=[]
C_full=[]
for X in C:
    C_left.append((X[0],X[1]))
    C_left.append((X[0],16-X[1]))
for X in C_left:
    C_full.append((X[0],X[1]))
    C_full.append((16-X[0],X[1]))
C_full = list(set(C_full))

    
N_left=[]
N_full=[]
for X in N:
    N_left.append((X[0],X[1]))
    N_left.append((X[0],16-X[1]))
for X in N_left:
    N_full.append((X[0],X[1]))
    N_full.append((16-X[0],X[1]))
N_full = list(set(N_full))

D_left=[]
D_full=[]
for X in D:
    D_left.append((X[0],X[1]))
    D_left.append((X[0],16-X[1]))
for X in D_left:
    D_full.append((X[0],X[1]))
    D_full.append((16-X[0],X[1]))   
D_full = list(set(D_full))


F_left=[]
F_full=[]
for X in F:
    F_left.append((X[0],X[1]))
    F_left.append((X[0],16-X[1]))
for X in F_left:
    F_full.append((X[0],X[1]))
    F_full.append((16-X[0],X[1]))    
F_full = list(set(F_full))

FD_left=[]
FD_full=[]
for X in FD:
    FD_left.append((X[0],X[1]))
    FD_left.append((X[0],16-X[1]))
for X in FD_left:
    FD_full.append((X[0],X[1]))
    FD_full.append((16-X[0],X[1]))    
FD_full = list(set(FD_full))


GT_left=[]
GT_full=[]
for X in GT:
    GT_left.append((X[0],X[1]))
    GT_left.append((X[0],16-X[1]))
for X in GT_left:
    GT_full.append((X[0],X[1]))
    GT_full.append((16-X[0],X[1]))    
GT_full = list(set(GT_full))

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:01:06 2023

@author: jean.ragusa
"""

import time as time
import copy
import numpy as np
from spydermesh import spydermesh
import matplotlib.pyplot as plt

plt.close("all")
# import matplotlib
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection


print("Running SPYDERMESH as the main code:")

plot_pins = False

color_list  = [
    "red",
    "yellow",
    "green",
    "blue",
    "pink",
    "cyan",
    "magenta",
    "blue",
    "white",
    "orange",
    "purple",
]

import matplotlib.colors as mcolors

# Convert the list to a set to remove duplicates
colors_set = set(color_list)

# Get a list of all named colors from matplotlib
all_named_colors = list(mcolors.CSS4_COLORS.keys())

# Add colors from all_named_colors to preferred_colors_set, avoiding duplicates
for color in all_named_colors:
    if color not in colors_set:
        colors_set.add(color)

# Convert the set back to a list
color_list = list(colors_set)

# shuffle the list of colors
# import random
# random.shuffle(color_list)

# %% global var
pitch = 0.63  # used to create that are alter deployed by x&y symetries
full_pitch = pitch * 2

# %%
def create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    mod_name,
    sectors,
    plot_pins,
):

    pin = spydermesh(pitch, pin_name)

    # polygonalize circles
    for R, n, hs, mat in zip(radii, nsub, half_list, mat_list):
        pin.polygonalize_circle(R, n, mat, half_shift=hs, preserve_vol=True)
    # add an extra circle in moderator
    pin.polygonalize_circle(
        rad_mod, nsub_mod, mod_name, half_shift=False, preserve_vol=False, stretch=0.35
    )

    # add a thin rectangular outer skin in moderator
    almost_pitch = np.max(pin.vert[-1][0])
    dp = pin.pitch - almost_pitch
    pin.add_corner_verts(mod_name, p=almost_pitch + dp / 2)
    # finish off moderator to fill the quarter pin pitch area
    pin.add_corner_verts(mod_name)

    # sectorization
    for iring, sector in enumerate(sectors):
        pin.add_sector_intersection(sector, iring)
    pin.collect_all_vertices()
    pin.make_polygons()

    pin.deploy_qpc()

    if plot_pins:
        uniq_mat, mat_id, mat_count = np.unique(
            pin.mat_poly, return_index=False, return_inverse=True, return_counts=True
        )
        colors = []
        for id_ in mat_id:
            colors.append(color_list[id_])
        pin.plot_polygons(colors=colors)
    return pin


# %%
def create_gt_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    mod_name,
    sectors,
    plot_pins,
):

    pin = spydermesh(pitch, pin_name)

    # polygonalize circles
    for R, n, hs, mat in zip(radii, nsub, half_list, mat_list):
        pin.polygonalize_circle(R, n, mat, half_shift=hs, preserve_vol=True)
    # add an extra circle in moderator
    pin.polygonalize_circle(
        rad_mod, nsub_mod, mod_name, half_shift=True, preserve_vol=False, stretch=0.5
    )

    # add a thin rectangular outer skin in moderator
    almost_pitch = np.max(pin.vert[-1][0])
    dp = pin.pitch - almost_pitch
    pin.add_corner_verts(mod_name, p=almost_pitch + dp / 2)
    # finish off moderator to fill the quarter pin pitch area
    pin.add_corner_verts(mod_name)

    # sectorization
    for iring, sector in enumerate(sectors):
        if iring == len(sectors) - 1:
            pin.add_sector_intersection(sector, iring, half_shift=False)
        else:
            pin.add_sector_intersection(sector, iring, half_shift=True)
    pin.collect_all_vertices()
    pin.make_polygons()

    pin.deploy_qpc()

    if plot_pins:
        uniq_mat, mat_id, mat_count = np.unique(
            pin.mat_poly, return_index=False, return_inverse=True, return_counts=True
        )
        colors = []
        for id_ in mat_id:
            colors.append(color_list[id_])
        pin.plot_polygons(colors=colors)
    return pin

# %%
class RectGrid:
    def __init__(self, pin_name, xlist, ylist, mat_name):
        self.name = pin_name
        self.xlist = xlist
        self.ylist = ylist
        self.vertices = self.create_vertices()
        self.polygons = self.create_cells()
        
        self.mat_poly = []
        for cell in self.polygons:
            self.mat_poly.append(mat_name)
        
        self.edge_vert_id = self.edge_id()

    def create_vertices(self):
        xx, yy = np.meshgrid(self.xlist, self.ylist)
        vertices = np.column_stack([xx.ravel(), yy.ravel()])
        return vertices

    def create_cells(self):
        num_x = len(self.xlist)
        num_y = len(self.ylist)
        cells = []

        for j in range(num_y - 1):
            for i in range(num_x - 1):
                # Calculate indices of the vertices of the current cell
                v0 = j * num_x + i
                v1 = v0 + 1
                v2 = v1 + num_x
                v3 = v0 + num_x
                cells.append([v0, v1, v2, v3, v0])

        return cells

    def edge_id(self):
        # IDs of the vertices that are on the periphery
        mask = np.zeros((len(self.vertices), 4), dtype=bool)
        extrema = np.zeros((2,2))
        for dim in range(2):
            extrema[0,dim] = np.min(self.vertices[:,dim])
            extrema[1,dim] = np.max(self.vertices[:,dim])
        counter = 0
        for dim in range(2):
            delta = np.abs(self.vertices[:, dim] - extrema[0,dim])
            mask_ = delta < 1e-9
            mask[:, counter] = mask_
            counter += 1
            delta = np.abs(self.vertices[:, dim] - extrema[1,dim])
            mask_ = delta < 1e-9
            mask[:, counter] = mask_
            counter += 1
        mask = np.logical_or.reduce(mask, axis=1)
        # get the indices where a vertex is along
        return np.where(mask)[0]  
# %%---- fuel pins




# gap size
half_water_gap = 0.04# 0.63*2
# compute the angles in [0,pi/4]
n_angles = 3
ang = np.linspace(0, np.pi / 4, n_angles)
ang = np.append(ang,-ang)
ang = np.unique(ang)
ang = np.sort(ang)
# compute the positions
pos_x = np.tan(ang)*pitch
pos_y = np.array([ -half_water_gap / 2, 0, half_water_gap / 2 ])
pos_y = np.array([ -half_water_gap / 2, half_water_gap / 2 ])


mod_name='water_outside'
water_gap_H = RectGrid('H', pos_x, pos_y, mod_name)
water_gap_V = RectGrid('V', pos_y, pos_x, mod_name)
water_gap_C = RectGrid('C', pos_y, pos_y, mod_name)

# %% inspect
# import inspect



# %%
# select a spyderweb pin using their name
def pick_pin(list_pins, name):
    for pin in list_pins:
        if pin.name == name:
            return copy.deepcopy(pin)
    raise ValueError("name {} not found in list of pins".format(name))


# %% lattice: empty spyderweb structure with the **full** pin pitch
lattice = spydermesh(full_pitch, "lat")



# %% select a specific lattice


# radii: 4 in fuel, one in gap, one in clad
radii = [0.13, 0.26, 0.39, 0.4096, 0.418, 0.475]
# angular subdivisions
nsub = [1, 1, 3, 3, 3, 3]
# whether polygons are rotated by half the angle spread
half_list = [False] * 6
# moderator zone
rad_mod = 0.5
nsub_mod = 3
mod_name = "water"
# sectorization
sectors = [0, 1, 1, 3, 3, 3, 3, 3, 3]

pin_name = "U"
# material names
mat_list = ["fuel_pincell", "fuel_pincell", "fuel_pincell", "fuel_pincell", "gap_pincell", "clad_pincell"]
uox = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_pincell",
    sectors,
    plot_pins,
)

pin_name = "CO"
# material names
mat_list = ["fuel_C", "fuel_C", "fuel_C", "fuel_C", "gap_C", "clad_C"]
uox_C = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_C",
    sectors,
    plot_pins,
)
pin_name = "E"
# material names
mat_list = ["fuel_E", "fuel_E", "fuel_E", "fuel_E", "gap_E", "clad_E"]
uox_E = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_E",
    sectors,
    plot_pins,
)

pin_name = "N"
# material names
mat_list = ["fuel_N", "fuel_N", "fuel_N", "fuel_N", "gap_N", "clad_N"]
uox_N = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_N",
    sectors,
    plot_pins,
)

pin_name = "D"
# material names
mat_list = ["fuel_D", "fuel_D", "fuel_D", "fuel_D", "gap_D", "clad_D"]
uox_D = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_D",
    sectors,
    plot_pins,
)

pin_name = "F"
# material names
mat_list = ["fuel_F", "fuel_F", "fuel_F", "fuel_F", "gap_F", "clad_F"]
uox_F = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_F",
    sectors,
    plot_pins,
)

pin_name = "FD"
# material names
mat_list = ["fuel_FD", "fuel_FD", "fuel_FD", "fuel_FD", "gap_FD", "clad_FD"]
uox_FD = create_fuel_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod,
    nsub_mod,
    "moderator_FD",
    sectors,
    plot_pins,
)



# radii = [0.13, 0.25, 0.41, 0.55, 0.59]
radii = [0.13, 0.25, 0.41, 0.561, 0.602]
nsub = [1, 1, 2, 2, 2]
half_list = [True] * 5
rad_mod_gt = 0.610
nsub_mod_gt = 2
sectors = [0, 1, 1, 2, 2, 2, 2, 3]

pin_name = "G"
mat_list = ["water_guide", "water_guide", "water_guide", "water_guide", "clad_guide"]
gtube = create_gt_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod_gt,
    nsub_mod_gt,
    'moderator_guide',
    sectors,
    plot_pins,
)

pin_name = "GT"
mat_list = ["water_GT", "water_GT", "water_GT", "water_GT", "clad_GT"]
gtube_GT = create_gt_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod_gt,
    nsub_mod_gt,
    'moderator_GT',
    sectors,
    plot_pins,
)


# radii = [0.13, 0.25, 0.41, 0.55, 0.59]
radii = [0.13, 0.25, 0.41, 0.559, 0.605]
nsub = [1, 1, 2, 2, 2]
half_list = [True] * 5
rad_mod_gt = 0.610
nsub_mod_gt = 2
mod_name_gt = "moderator_instru"
sectors = [0, 1, 1, 2, 2, 2, 2, 3]

pin_name = "I"
mat_list = ["water_instru", "water_instru", "water_instru", "water_instru", "clad_instru"]
ginstru = create_gt_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod_gt,
    nsub_mod_gt,
    'moderator_instru',
    sectors,
    plot_pins,
)

pin_name = "IT"
mat_list = ["water_IT", "water_IT", "water_IT", "water_IT", "clad_IT"]
ginstru_IT = create_gt_pin(
    pin_name,
    radii,
    nsub,
    half_list,
    mat_list,
    rad_mod_gt,
    nsub_mod_gt,
    'moderator_IT',
    sectors,
    plot_pins,
)




# list all of the possible pin types that were created
list_pins = [uox,gtube,ginstru, water_gap_C, water_gap_H, water_gap_V,uox_C,uox_F,uox_E,uox_N,uox_D,uox_FD,ginstru_IT,gtube_GT]
casename = "17x17_6fam"


if casename == "17x17_1fam":
    lat = np.empty((19, 19), dtype="<U3")
    lat[:, :] = "U"
    gt = []
    it = []
    gt.append([2, 5])
    gt.append([2, 8])
    gt.append([2, 11])
    gt.append([3, 3])
    gt.append([3, 13])
    gt.append([5, 2])
    gt.append([5, 5])
    gt.append([5, 8])
    gt.append([5, 11])
    gt.append([5, 14])
    gt.append([8, 2])
    gt.append([8, 5])
    it.append([8, 8])
    gt.append([8, 11])
    gt.append([8, 14])
    gt.append([11, 2])
    gt.append([11, 5])
    gt.append([11, 8])
    gt.append([11, 11])
    gt.append([11, 14])
    gt.append([13, 3])
    gt.append([13, 13])
    gt.append([14, 5])
    gt.append([14, 8])
    gt.append([14, 11])
    for ij in gt:
        i, j = ij[0]+1, ij[1]+1
        lat[i, j] = "G"
    for ij in it:
        i, j = ij[0]+1, ij[1]+1
        lat[i, j] = "I"        
    lat[:, 0] = "V"
    lat[:, -1] = "V"
    lat[0, :] = "H"
    lat[-1, :] = "H"
    lat[0, 0] = "C"
    lat[0, -1] = "C"
    lat[-1, 0] = "C"
    lat[-1, -1] = "C"
    print("casename=",casename,"\n",lat)


elif casename == "17x17_6fam":
    lat = np.empty((19, 19), dtype="<U5")
    lat[:, :] = "U"
    for ij in E_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "E"
    for ij in C_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "CO"
    for ij in N_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "N"
    for ij in D_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "D"
    for ij in F_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "F"
    for ij in FD_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "FD"
    for ij in GT_full:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "GT"
    for ij in IT:
        i, j = ij[0], ij[1]
        lat[i+1, j+1] = "IT"        
    lat[:, 0] = "V"
    lat[:, -1] = "V"
    lat[0, :] = "H"
    lat[-1, :] = "H"
    lat[0, 0] = "C"
    lat[0, -1] = "C"
    lat[-1, 0] = "C"
    lat[-1, -1] = "C"
    print("casename=",casename,"\n",lat)



elif casename == "C5G7_2x2":
    latU = np.empty((17, 17), dtype="<U1")
    latU[:, :] = "U"
    gt = []
    gt.append([2, 5])
    gt.append([2, 8])
    gt.append([2, 11])
    gt.append([3, 3])
    gt.append([3, 13])
    gt.append([5, 2])
    gt.append([5, 5])
    gt.append([5, 8])
    gt.append([5, 11])
    gt.append([5, 14])
    gt.append([8, 2])
    gt.append([8, 5])
    gt.append([8, 8])
    gt.append([8, 11])
    gt.append([8, 14])
    gt.append([11, 2])
    gt.append([11, 5])
    gt.append([11, 8])
    gt.append([11, 11])
    gt.append([11, 14])
    gt.append([13, 3])
    gt.append([13, 13])
    gt.append([14, 5])
    gt.append([14, 8])
    gt.append([14, 11])
    for ij in gt:
        i, j = ij[0], ij[1]
        latU[i, j] = "G"
    print("casename=",casename,"\n",latU)

    latM = np.empty((17, 17), dtype="<U1")
    latM[:, :] = "1"
    latM[0, :] = "3"
    latM[-1, :] = "3"
    latM[:, 0] = "3"
    latM[:, -1] = "3"
    latM[1:3, 1:-1] = "2"
    latM[-3:-1, 1:-1] = "2"
    latM[1:-1, 1:3] = "2"
    latM[1:-1, -3:-1] = "2"
    latM[3, 4] = "2"
    latM[-4, 4] = "2"
    latM[4, 3] = "2"
    latM[-5, 3] = "2"
    latM[3, -5] = "2"
    latM[-4, -5] = "2"
    latM[4, -4] = "2"
    latM[-5, -4] = "2"

    for ij in gt:
        i, j = ij[0], ij[1]
        latM[i, j] = "G"
    print("casename=",casename,"\n",latM)

    lat = np.vstack( ( np.hstack((latU, latM)), np.hstack((latM, latU)) ) )
    # lat = np.vstack( ( np.hstack((latU, latM)), np.hstack((latM, latU)), np.hstack((latU, latM)) ) )

elif casename == "C5G7_2x2_water_gap":
    latU = np.empty((19, 19), dtype="<U1")
    latU[:, :] = "U"
    gt = []
    gt.append([2, 5])
    gt.append([2, 8])
    gt.append([2, 11])
    gt.append([3, 3])
    gt.append([3, 13])
    gt.append([5, 2])
    gt.append([5, 5])
    gt.append([5, 8])
    gt.append([5, 11])
    gt.append([5, 14])
    gt.append([8, 2])
    gt.append([8, 5])
    gt.append([8, 8])
    gt.append([8, 11])
    gt.append([8, 14])
    gt.append([11, 2])
    gt.append([11, 5])
    gt.append([11, 8])
    gt.append([11, 11])
    gt.append([11, 14])
    gt.append([13, 3])
    gt.append([13, 13])
    gt.append([14, 5])
    gt.append([14, 8])
    gt.append([14, 11])
    for ij in gt:
        i, j = ij[0]+1, ij[1]+1
        latU[i, j] = "G"
    latU[:, 0] = "V"
    latU[:, -1] = "V"
    latU[0, :] = "H"
    latU[-1, :] = "H"
    latU[0, 0] = "C"
    latU[0, -1] = "C"
    latU[-1, 0] = "C"
    latU[-1, -1] = "C"
    print("casename=",casename,"\n",latU)

    latM = np.empty((19, 19), dtype="<U1")
    latM[:, :] = "1"
    latM[1, :] = "3"
    latM[-2, :] = "3"
    latM[:, 1] = "3"
    latM[:, -2] = "3"
    latM[2:4, 2:-2] = "2"
    latM[-4:-2, 2:-2] = "2"
    latM[2:-2, 2:4] = "2"
    latM[2:-2, -4:-2] = "2"
    latM[4, 5] = "2"
    latM[-5, 5] = "2"
    latM[5, 4] = "2"
    latM[-6, 4] = "2"
    latM[4, -6] = "2"
    latM[-5, -6] = "2"
    latM[5, -5] = "2"
    latM[-6, -5] = "2"

    for ij in gt:
        i, j = ij[0]+1, ij[1]+1
        latM[i, j] = "G"
    latM[:, 0] = "V"
    latM[:, -1] = "V"
    latM[0, :] = "H"
    latM[-1, :] = "H"
    latM[0, 0] = "C"
    latM[0, -1] = "C"
    latM[-1, 0] = "C"
    latM[-1, -1] = "C"
    print("casename=",casename,"\n",latM)

    lat = np.vstack( ( np.hstack((latU, latM)), np.hstack((latM, latU)) ) )
    # lat = np.vstack( ( np.hstack((latU, latM)), np.hstack((latM, latU)), np.hstack((latU, latM)) ) )

else:
    raise ValueError("casename {} not recognized".format(casename))
# %% put together the lattice

nx, ny = lat.shape

dx_prev, dy_prev = 0., 0.

for i in range(nx):
    if i == 0:
        first_row = True
        delta_y = 0.
    else:
        first_row = False
        
    for j in range(ny):
        
        pin = pick_pin(list_pins, lat[i, j])
        # print(fuel.polygons[-1])
        # print(pin.polygons[-1])
        pt_min = np.min(pin.vertices, axis=0)
        pt_max = np.max(pin.vertices, axis=0)
        dx, dy = pt_max - pt_min

        if j == 0:
            first_col = True
            delta_x = 0.
        else:
            first_col = False

        if first_row and first_col:
            lattice.nverts = len(pin.vertices)
            lattice.vertices = np.copy(pin.vertices)
            lattice.polygons = pin.polygons.copy()
            lattice.mat_poly = pin.mat_poly.copy()
            lattice.edge_vert_id = pin.edge_vert_id.copy()
        else:
            # update vertex id's
            poly_pin = pin.polygons.copy()
            for ip, p in enumerate(poly_pin):
                for iv, vid in enumerate(p):
                    pin.polygons[ip][iv] += lattice.nverts
            # update list of polygons
            lattice.polygons += pin.polygons
            # shift vertex locations
            new_verts = np.copy(pin.vertices)
            # print("delta_x=", dx_prev_2 + dx_cur_2,", delta_y=", dy_prev_2 + dy_cur_2,"\n")
            # update skip in x starting at second column
            if j > 0 : 
                delta_x += dx_prev/2 + dx/2
            # update skip in y starting at second row
            if j == 0 and i > 0 : 
                delta_y += dy_prev/2 + dy/2
                
            new_verts[:, 0] += delta_x 
            new_verts[:, 1] -= delta_y 
            # update vertex coordinate array
            lattice.vertices = np.vstack((lattice.vertices, new_verts))
            # update polygon names
            lattice.mat_poly += pin.mat_poly
            # update indices of vertices that live on the periphery of a pin cell
            edge_vert_id = pin.edge_vert_id[:] + lattice.nverts
            lattice.edge_vert_id = np.hstack((lattice.edge_vert_id, edge_vert_id))
            # update # of vertices so far
            lattice.nverts += len(pin.vertices)
        
        # save previous cell sizes
        dx_prev, dy_prev = dx, dy


print(lattice.vertices.shape)

# new1_lattice = copy.deepcopy(lattice)
# new1_lattice.make_vertices_unique()
# print(new1_lattice.vertices.shape)

# t0 = time.time()
# new1_lattice = copy.deepcopy(lattice)
# new1_lattice.make_vertices_unique2()
# print('elapsed time =',time.time()-t0)
# print(new1_lattice.vertices.shape)

t0 = time.time()
lattice.make_vertices_unique3()
print('elapsed time =',time.time()-t0)
print(lattice.vertices.shape)


# %%
plot_lattice = True
if plot_lattice:
    uniq_mat, mat_id, mat_count = np.unique(
        lattice.mat_poly, return_index=False, return_inverse=True, return_counts=True
    )
    colors = []
    for id_ in mat_id:
        colors.append(color_list[id_])
    lattice.plot_polygons(colors=colors, size_=.1, lw_=0.2)
    
# %%
lattice.export_to_obj("lattice_{}.obj".format(casename))

# %% verif area
pt_min = np.min(lattice.vertices, axis=0)
pt_max = np.max(lattice.vertices, axis=0)
dx, dy = pt_max - pt_min
A_truth = dx*dy

A_sum = 0.0
for i, poly in enumerate(lattice.polygons):
    # print(i,poly)
    coord = lattice.vertices[poly]
    A = lattice.PolyArea_noabs(coord[:, 0], coord[:, 1])
    if A < 0:
        print("A<0", poly, coord, A)
    A_sum += A
print("Asum error=", A_sum - A_truth)  
print(mat_id,len(mat_id))