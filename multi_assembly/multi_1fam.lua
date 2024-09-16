--- Import mesh

my_filename = "lattice_3x3_1fam.obj"

meshgen1 = mesh.MeshGenerator.Create
({
  inputs =
  {
    mesh.FromFileMeshGenerator.Create
    ({
      filename = my_filename
    })
  }
})
mesh.MeshGenerator.Execute(meshgen1)
mesh.ExportToPVTU("mesh_1fam")
-- mesh.ComputeVolumePerMaterialID()
-- os.exit()

-- Create cross sections


mat_names = {
    {'B4C_b4c_1_family', '2H'},
    {'clad_2B_guide', '2B'},
    {'clad_2B_instru', '2B'},
    {'clad_2B_pincell', '2B'},
    {'clad_b4c_1_family', '2H'},
    {'clad_guide_instru_1_family', '2H'},
    {'clad_pincell_1_family', '2H'},
    {'fuel_2B_pincell', '2B'},
    {'fuel_pincell_1_family', '2H'},
    {'gap_2B_pincell', '2B'},
    {'gap_b4c_1_family', '2H'},
    {'gap_pincell_1_family', '2H'},
    {'guide_b4c_1_family', '2H'},
    {'moderator_2B_guide', '2B'},
    {'moderator_2B_instru', '2B'},
    {'moderator_2B_pincell', '2B'},
    {'moderator_b4c_1_family', '2H'},
    {'moderator_instru_1_family', '2H'},
    {'moderator_pincell_1_family', '2H'},
    {'water_2B_guide', '2B'},
    {'water_2B_instru', '2B'},
    {'water_guide_instru_1_family', '2H'},
    {'water_outside_B', '2B'},
    {'water_outside_H', '2H'}
}
-- materials = {}
-- my_xs = {}
-- xs_file_2B = 'mgxs_2A_bis.h5'
-- xs_file_2H = 'mgxs_2H.h5'
-- Nmat = #mat_names

-- for imat = 1, Nmat do
--     my_xs[mat_names[imat]] = xs.Create()
--     xs.Set(my_xs[mat_names[imat]],OPENMC_XSLIB,xs_file,294.0,mat_names[imat])
--     materials[imat] = mat.AddMaterial(mat_names[imat])
--     mat.SetProperty(materials[imat], TRANSPORT_XSECTIONS, OPENMC_XSLIB, xs_file, 294, mat_names[imat])
-- end

materials = {}
my_xs = {}
xs_files = {
    ['2B'] = 'mgxs_2A_bis.h5',
    ['2H'] = 'mgxs_2H.h5'
}

Nmat = #mat_names

for imat = 1, Nmat do
    local mat_name = mat_names[imat][1]
    local xs_key = mat_names[imat][2]
    local xs_file = xs_files[xs_key]

    my_xs[mat_name] = xs.Create()  -- Utiliser la première valeur comme clé
    xs.Set(my_xs[mat_name], OPENMC_XSLIB, xs_file, 294.0, mat_name)
    
    materials[imat] = mat.AddMaterial(mat_name)
    mat.SetProperty(materials[imat], TRANSPORT_XSECTIONS, OPENMC_XSLIB, xs_file, 294, mat_name)
end






-- Setup Physics

-- Angular quadrature
pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV,2,1)
aquad.OptimizeForPolarSymmetry(pquad, 4.0*math.pi)

-- Solver
num_groups = 361

--############################################### Setup Physics
lbs_block = {
  num_groups = num_groups,
  groupsets = {
    {
      groups_from_to = { 0, num_groups - 1 },
      angular_quadrature_handle = pquad,
      inner_linear_method = "krylov_richardson",
      l_max_its = 20,
      l_abs_tol = 1e-6,
      angle_aggregation_type = "polar",
    },
  },
}

lbs_options = {
    boundary_conditions = {
        { name = "xmin", type = "reflecting" },
        { name = "xmax", type = "reflecting" },
        { name = "ymin", type = "reflecting" },
        { name = "ymax", type = "reflecting" },
        { name = "zmin", type = "reflecting" },
        { name = "zmax", type = "reflecting" },

      },
    scattering_order = 3,
    verbose_inner_iterations = true,
    verbose_outer_iterations = true,
    -- power_field_function_on = true,
    power_default_kappa = 1.0,
    power_normalization = 1.0,
    save_angular_flux = false,
    write_restart_time_interval = 3660,
    write_restart_path = "1fam_restart/1fam",       
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

k_solver = lbs.PowerIterationKEigen.Create({
  lbs_solver_handle = phys1,
  k_tol = 1.0e-8,
  -- max_iters = 5,
})
-- k_solver = lbs.PowerIterationKEigenSMM.Create({
--     lbs_solver_handle = phys1,
--     accel_pi_verbose = true,
--     k_tol = 1.0e-8,
--     accel_pi_k_tol = 1.0e-8,
--     accel_pi_max_its = 30,
--     diff_sdm = "pwld",
-- })
-- k_solver = lbs.PowerIterationKEigenSCDSA.Create({
--     lbs_solver_handle = phys1,
--     diff_accel_sdm = "pwld",
--     accel_pi_verbose = true,
--     k_tol = 1.0e-8,
--     accel_pi_k_tol = 1.0e-8,
--     accel_pi_max_its = 30,
-- })
solver.Initialize(k_solver)
solver.Execute(k_solver)



fflist, count = lbs.GetScalarFieldFunctionList(phys1)

for key, value in pairs(fflist) do
    print(key, type(value))
end

if location_id == 0 then
    print('Longueur de la table est :', #fflist, count)
end
MPIBarrier()
log.Log(LOG_0, 'Longueur de la table est (v2):'..#fflist)

-- os.exit()

-- if fflist == nil then
--   print("fflist est nil")
-- else
--   for _, value in ipairs(fflist) do
--       print(value)
--   end
-- end

-- Define the pitch
local pitch = 1.26

-- Define the number of cells
local num_cells = 17

-- Function to compute the center of a cell
local function compute_cell_center(i, j, pitch, x_ref, y_ref)
    local x_center = ((i - 1) * pitch - (num_cells // 2) * pitch) + x_ref
    local y_center = ((j - 1) * pitch - (num_cells // 2) * pitch) + y_ref
    return x_center, y_center
end

-- Define the pairs to avoid
local avoid_pairs = {
    [4] = {14, 4},
    [12] = {3, 6, 9, 15, 12},
    [3] = {6, 12, 9},
    [9] = {3, 6, 12, 15, 9},
    [14] = {14, 4},
    [15] = {6, 12, 9},
    [6] = {3, 6, 12, 9, 15}
}

-- Function to check if a pair should be avoided
local function should_avoid(i, j)
    if avoid_pairs[i] then
        for _, value in ipairs(avoid_pairs[i]) do
            if value == j then
                return true
            end
        end
    end
    return false
end


-- Initialize table to store values
local val_table = {}

-- Loop over the cells
-- Définir les paramètres
num_cells = 17  -- Taille de l'assemblage
size = 17  -- Taille de l'assemblage
pitch = 1.26  -- Pitch entre les assemblages
dr = 0.04  -- Décalage


-- Position de référence pour le premier assemblage en haut à gauche

-- Créer la table `val_C1` pour le premier assemblage en haut à gauche
local val_C1 = {}
for i = 1, num_cells do
    val_C1[i] = {}  -- Initialiser la table interne pour chaque ligne
    for j = 1, num_cells do
        val_C1[i][j] = -1  -- Valeur par défaut
        if not should_avoid(i, j) then

            -- Calculer les coordonnées du centre de la cellule pour cet assemblage
            x_ref, y_ref = -(size * pitch + dr * 2), (size * pitch + dr * 2)
            x_center, y_center = compute_cell_center(i, j, pitch, x_ref, y_ref)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })
            
            -- Obtenir les données de la cellule
            xs_fuel_pincell = xs.Get(my_xs["fuel_2B_pincell"])
    
            -- Vérifier si 'sigma_f' est valide
            if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
                sig_f = xs_fuel_pincell["sigma_f"]
            else
                error("sigma_f is not a valid number or does not exist in the table")
            end
    
            -- Calculer la valeur pour la cellule
            val = 0.
            for g = 1, num_groups do
                ffi = fieldfunc.FFInterpolationCreate(VOLUME)
                fieldfunc.SetProperty(ffi, OPERATION, OP_SUM)
                fieldfunc.SetProperty(ffi, LOGICAL_VOLUME, my_lv)
                fieldfunc.SetProperty(ffi, ADD_FIELDFUNCTION, fflist[g])
                
                fieldfunc.Initialize(ffi)
                fieldfunc.Execute(ffi)
                val_g = fieldfunc.GetValue(ffi)
                val = val + val_g * sig_f[g]
            end
            val_C1[i][j] = val
        end     
    end
end   


-- Print the table values (optional, for verification)
if location_id == 0 then
  ofile = io.open('power_C.txt',"w")
  io.output(ofile)  
  for i = 1, num_cells do
    for j = 1, num_cells do 
        io.write(string.format("val_C[%d][%d] = %.5f\n", i, j, val_C1[i][j]))
    end
  end
  io.close(ofile)  
end

local val_E1 = {}
for i = 1, num_cells do
    val_E1[i] = {}  -- Initialiser la table interne pour chaque ligne
    for j = 1, num_cells do
        val_E1[i][j] = -1  -- Valeur par défaut
        if not should_avoid(i, j) then

            -- Calculer les coordonnées du centre de la cellule pour cet assemblage
            x_ref, y_ref = 0, (size * pitch + dr * 2)
            x_center, y_center = compute_cell_center(i, j, pitch, x_ref, y_ref)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })
            
            -- Obtenir les données de la cellule
            xs_fuel_pincell = xs.Get(my_xs["fuel_2B_pincell"])
    
            -- Vérifier si 'sigma_f' est valide
            if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
                sig_f = xs_fuel_pincell["sigma_f"]
            else
                error("sigma_f is not a valid number or does not exist in the table")
            end
    
            -- Calculer la valeur pour la cellule
            val = 0.
            for g = 1, num_groups do
                ffi = fieldfunc.FFInterpolationCreate(VOLUME)
                fieldfunc.SetProperty(ffi, OPERATION, OP_SUM)
                fieldfunc.SetProperty(ffi, LOGICAL_VOLUME, my_lv)
                fieldfunc.SetProperty(ffi, ADD_FIELDFUNCTION, fflist[g])
                
                fieldfunc.Initialize(ffi)
                fieldfunc.Execute(ffi)
                val_g = fieldfunc.GetValue(ffi)
                val = val + val_g * sig_f[g]
            end
            val_E1[i][j] = val
        end     
    end
end 


-- Print the table values (optional, for verification)
if location_id == 0 then
  ofile = io.open('power_E1.txt',"w")
  io.output(ofile)  
  for i = 1, num_cells do
    for j = 1, num_cells do 
        io.write(string.format("val_E1[%d][%d] = %.5f\n", i, j, val_E1[i][j]))
    end
  end
  io.close(ofile)  
end

local val_E2 = {}
for i = 1, num_cells do
    val_E2[i] = {}  -- Initialiser la table interne pour chaque ligne
    for j = 1, num_cells do
        val_E2[i][j] = -1  -- Valeur par défaut
        if not should_avoid(i, j) then

            -- Calculer les coordonnées du centre de la cellule pour cet assemblage
            x_ref, y_ref = -(size * pitch + dr * 2), 0
            x_center, y_center = compute_cell_center(i, j, pitch, x_ref, y_ref)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })
            
            -- Obtenir les données de la cellule
            xs_fuel_pincell = xs.Get(my_xs["fuel_2B_pincell"])
    
            -- Vérifier si 'sigma_f' est valide
            if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
                sig_f = xs_fuel_pincell["sigma_f"]
            else
                error("sigma_f is not a valid number or does not exist in the table")
            end
    
            -- Calculer la valeur pour la cellule
            val = 0.
            for g = 1, num_groups do
                ffi = fieldfunc.FFInterpolationCreate(VOLUME)
                fieldfunc.SetProperty(ffi, OPERATION, OP_SUM)
                fieldfunc.SetProperty(ffi, LOGICAL_VOLUME, my_lv)
                fieldfunc.SetProperty(ffi, ADD_FIELDFUNCTION, fflist[g])
                
                fieldfunc.Initialize(ffi)
                fieldfunc.Execute(ffi)
                val_g = fieldfunc.GetValue(ffi)
                val = val + val_g * sig_f[g]
            end
            val_E2[i][j] = val
        end     
    end
end 


-- Print the table values (optional, for verification)
if location_id == 0 then
  ofile = io.open('power_E2.txt',"w")
  io.output(ofile)  
  for i = 1, num_cells do
    for j = 1, num_cells do 
        io.write(string.format("val_E1[%d][%d] = %.5f\n", i, j, val_E2[i][j]))
    end
  end
  io.close(ofile)  
end

local val_M = {}
for i = 1, num_cells do
    val_M[i] = {}  -- Initialiser la table interne pour chaque ligne
    for j = 1, num_cells do
        val_M[i][j] = -1  -- Valeur par défaut
        if not should_avoid(i, j) then

            -- Calculer les coordonnées du centre de la cellule pour cet assemblage
            x_ref, y_ref = 0, 0
            x_center, y_center = compute_cell_center(i, j, pitch, x_ref, y_ref)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })
            
            -- Obtenir les données de la cellule
            xs_fuel_pincell = xs.Get(my_xs["fuel_pincell_1_family"])
    
            -- Vérifier si 'sigma_f' est valide
            if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
                sig_f = xs_fuel_pincell["sigma_f"]
            else
                error("sigma_f is not a valid number or does not exist in the table")
            end
    
            -- Calculer la valeur pour la cellule
            val = 0.
            for g = 1, num_groups do
                ffi = fieldfunc.FFInterpolationCreate(VOLUME)
                fieldfunc.SetProperty(ffi, OPERATION, OP_SUM)
                fieldfunc.SetProperty(ffi, LOGICAL_VOLUME, my_lv)
                fieldfunc.SetProperty(ffi, ADD_FIELDFUNCTION, fflist[g])
                
                fieldfunc.Initialize(ffi)
                fieldfunc.Execute(ffi)
                val_g = fieldfunc.GetValue(ffi)
                val = val + val_g * sig_f[g]
            end
            val_M[i][j] = val
        end     
    end
end 


-- Print the table values (optional, for verification)
if location_id == 0 then
  ofile = io.open('power_M.txt',"w")
  io.output(ofile)  
  for i = 1, num_cells do
    for j = 1, num_cells do 
        io.write(string.format("val_EM[%d][%d] = %.5f\n", i, j, val_M[i][j]))
    end
  end
  io.close(ofile)  
end



MPIBarrier()
log.Log(LOG_0,'ici')
fieldfunc.ExportToVTKMulti(fflist,'flux_1fam')


