d = {
    ifba = {
        {5, 1},
        {8, 9},
        {10, 6},
        {9, 8},
        {2, 2},
        {15, 5},
        {4, 2},
        {5, 3},
        {5, 12},
        {11, 7},
        {2, 4},
        {0, 16},
        {1, 8},
        {8, 4},
        {0, 0},
        {11, 9},
        {13, 12},
        {6, 6},
        {7, 5},
        {7, 14},
        {12, 13},
        {5, 7},
        {9, 5},
        {9, 14},
        {13, 5},
        {15, 11},
        {14, 12},
        {3, 12},
        {4, 11},
        {5, 9},
        {11, 4},
        {1, 5},
        {12, 8},
        {3, 5},
        {14, 14},
        {4, 13},
        {10, 10},
        {7, 2},
        {7, 11},
        {14, 7},
        {5, 4},
        {9, 2},
        {5, 13},
        {8, 15},
        {13, 11},
        {16, 16},
        {12, 3},
        {14, 9},
        {4, 8},
        {13, 4},
        {16, 0},
        {1, 11},
        {11, 13},
        {6, 10},
        {12, 5},
        {14, 2},
        {12, 14},
        {3, 11},
        {8, 1},
        {11, 15},
        {2, 12},
        {14, 4},
        {3, 4},
        {4, 3},
        {8, 12},
        {9, 11},
        {2, 14},
        {15, 8},
        {4, 5},
        {4, 14},
        {11, 1},
        {5, 15},
        {2, 7},
        {12, 2},
        {12, 11},
        {8, 7},
        {11, 3},
        {11, 12},
        {2, 9},
        {7, 8}
    }
}
--- Import mesh

my_filename = "lattice_2L.obj"

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
mesh.ExportToPVTU("17x17")
-- mesh.ComputeVolumePerMaterialID()
-- os.exit()

-- Create cross sections


mat_names = {'clad_guide_guide', 'clad_guide_instru', 'clad_ifba', 'clad_normal', 'coat_ifba', 'fuel_ifba', 'fuel_normal', 'gap_ifba', 'gap_normal', 'moderator_guide', 'moderator_ifba', 'moderator_instru', 'moderator_normal', 'water_guide_guide', 'water_guide_instru', 'water_outside'}
materials = {}
my_xs = {}
xs_file = 'mgxs_2L.h5'

Nmat = #mat_names

for imat = 1, Nmat do
    my_xs[mat_names[imat]] = xs.Create()
    xs.Set(my_xs[mat_names[imat]],OPENMC_XSLIB,xs_file,294.0,mat_names[imat])
    materials[imat] = mat.AddMaterial(mat_names[imat])
    mat.SetProperty(materials[imat], TRANSPORT_XSECTIONS, OPENMC_XSLIB, xs_file, 294, mat_names[imat])
end




-- Setup Physics

-- Angular quadrature
pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV,32,4)
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
    write_restart_path = "2L_restart/2L",
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

k_solver = lbs.PowerIterationKEigen.Create({
  lbs_solver_handle = phys1,
  k_tol = 1.0e-8,
  -- max_iters = 1,
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

-- for key, value in pairs(fflist) do
--     print(key, type(value))
-- end

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
local function compute_cell_center(i, j, pitch)
    local x_center = (i - 1) * pitch - (num_cells // 2) * pitch
    local y_center = (j - 1) * pitch - (num_cells // 2) * pitch
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
for i = 1, num_cells do
    val_table[i] = {} -- initialize inner table for each row
    for j = 1, num_cells do
        val_table[i][j] = -1
        if not should_avoid(i, j) then
            x_center, y_center = compute_cell_center(i, j, pitch)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })

            -- Initialisation de family_name par défaut
            local family_name = 'fuel_normal'
            
            -- Vérification si la position est dans la liste de 'gado'
            for _, pos in ipairs(d['ifba']) do
                if pos[1] == (i - 1) and pos[2] == (j - 1) then
                    family_name = 'fuel_ifba'
                    break
                end
            end

            -- Utiliser 'fuel_famille' (par exemple 'fuel_gado') au lieu de 'fuel_pincell'
            xs_fuel_pincell = xs.Get(my_xs[family_name])
            if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
                sig_f = xs_fuel_pincell["sigma_f"]
            else
                error("sigma_f is not a valid number or does not exist in the table")
            end

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
            val_table[i][j] = val
        end
    end
end



-- Print the table values (optional, for verification)
if location_id == 0 then
  ofile = io.open('power_2L.txt',"w")
  io.output(ofile)  
  for i = 1, num_cells do
    for j = 1, num_cells do 
        io.write(string.format("val_table[%d][%d] = %.5f\n", i, j, val_table[i][j]))
    end
  end
  io.close(ofile)  
end

MPIBarrier()
log.Log(LOG_0,'ici')
-- fieldfunc.ExportToVTKMulti(fflist,'flux_2J')