--- Import mesh

d = {
    C = {{0, 16}, {16, 16}, {16, 0}, {0, 0}},
    D = {{12, 4}, {4, 9}, {10, 6}, {2, 2}, {7, 1}, {7, 10}, {3, 6}, {9, 1}, {9, 10}, {13, 10}, {15, 7}, {6, 4}, {7, 3}, {6, 13}, {7, 12}, {9, 3}, {9, 12}, {1, 10}, {15, 9}, {6, 6}, {3, 10}, {10, 15}, {1, 12}, {7, 7}, {12, 6}, {12, 15}, {9, 7}, {13, 7}, {15, 4}, {6, 1}, {7, 9}, {4, 4}, {14, 14}, {9, 9}, {10, 1}, {10, 10}, {1, 7}, {15, 6}, {13, 9}, {12, 1}, {6, 15}, {12, 10}, {3, 7}, {4, 6}, {4, 15}, {10, 3}, {10, 12}, {1, 9}, {7, 4}, {12, 12}, {3, 9}, {6, 10}, {14, 2}, {4, 1}, {4, 10}, {10, 7}, {1, 4}, {6, 3}, {6, 12}, {12, 7}, {4, 12}, {10, 9}, {1, 6}, {2, 14}, {7, 13}, {12, 9}, {9, 4}, {9, 13}, {15, 10}, {6, 7}, {7, 6}, {7, 15}, {4, 7}, {10, 4}, {9, 6}, {10, 13}, {9, 15}, {13, 6}, {15, 12}, {6, 9}},
    F = {{5, 1}, {14, 13}, {5, 10}, {8, 9}, {9, 8}, {13, 8}, {15, 5}, {6, 2}, {6, 11}, {5, 3}, {5, 12}, {11, 7}, {1, 8}, {3, 8}, {8, 4}, {11, 9}, {7, 5}, {7, 14}, {14, 10}, {5, 7}, {9, 5}, {9, 14}, {13, 5}, {13, 14}, {15, 11}, {14, 3}, {5, 9}, {4, 11}, {11, 4}, {10, 8}, {1, 5}, {2, 13}, {12, 8}, {3, 5}, {3, 14}, {8, 13}, {2, 6}, {7, 2}, {7, 11}, {14, 7}, {5, 4}, {9, 2}, {5, 13}, {8, 6}, {13, 2}, {8, 15}, {13, 11}, {6, 8}, {14, 9}, {5, 6}, {4, 8}, {10, 5}, {10, 14}, {1, 11}, {11, 13}, {2, 10}, {12, 5}, {3, 2}, {3, 11}, {8, 1}, {8, 10}, {11, 6}, {2, 3}, {11, 15}, {8, 3}, {8, 12}, {9, 11}, {15, 8}, {6, 5}, {6, 14}, {14, 6}, {4, 5}, {10, 2}, {11, 1}, {5, 15}, {10, 11}, {11, 10}, {2, 7}, {12, 11}, {8, 7}, {11, 3}, {11, 12}, {2, 9}, {7, 8}},
    N = {{14, 1}, {3, 1}, {1, 3}, {15, 2}, {1, 15}, {15, 14}, {3, 15}, {14, 15}, {1, 2}, {2, 1}, {13, 1}, {15, 1}, {1, 14}, {15, 13}, {1, 1}, {1, 13}, {15, 3}, {2, 15}, {13, 15}, {15, 15}},
    FD = {{2, 4}, {13, 4}, {14, 4}, {4, 13}, {4, 3}, {3, 4}, {12, 13}, {4, 2}, {12, 3}, {3, 12}, {12, 2}, {2, 12}, {4, 14}, {14, 12}, {13, 12}, {12, 14}},
    E = {{4, 0}, {12, 16}, {3, 16}, {14, 16}, {8, 0}, {0, 2}, {5, 16}, {10, 0}, {0, 5}, {1, 0}, {0, 8}, {0, 14}, {16, 4}, {0, 11}, {16, 1}, {16, 7}, {16, 10}, {16, 13}, {12, 0}, {7, 16}, {14, 0}, {3, 0}, {5, 0}, {0, 1}, {9, 16}, {0, 7}, {0, 4}, {0, 10}, {11, 16}, {0, 13}, {16, 3}, {2, 16}, {13, 16}, {7, 0}, {16, 6}, {16, 12}, {16, 9}, {15, 16}, {16, 15}, {6, 16}, {4, 16}, {6, 0}, {9, 0}, {11, 0}, {0, 3}, {2, 0}, {13, 0}, {0, 9}, {0, 6}, {0, 12}, {16, 2}, {15, 0}, {10, 16}, {0, 15}, {16, 5}, {8, 16}, {1, 16}, {16, 8}, {16, 14}, {16, 11}},
    GT = {{3, 13}, {11, 2}, {11, 5}, {11, 8}, {2, 5}, {11, 14}, {2, 11}, {11, 11}, {2, 8}, {3, 3}, {8, 2}, {8, 5}, {8, 11}, {8, 14}, {13, 13}, {14, 5}, {14, 11}, {5, 2}, {14, 8}, {5, 5}, {5, 11}, {5, 8}, {5, 14}, {13, 3}},
    IT = {{8, 8}}
}

my_filename = "lattice_17x17_6fam.obj"

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
mesh.ExportToPVTU("mesh_2A_6")
-- mesh.ComputeVolumePerMaterialID()
-- os.exit()

-- Create cross sections


mat_names = {'clad_C',
 'clad_D',
 'clad_E',
 'clad_F',
 'clad_FD',
 'clad_GT',
 'clad_IT',
 'clad_N',
 'fuel_C',
 'fuel_D',
 'fuel_E',
 'fuel_F',
 'fuel_FD',
 'fuel_N',
 'gap_C',
 'gap_D',
 'gap_E',
 'gap_F',
 'gap_FD',
 'gap_N',
 'moderator_C',
 'moderator_D',
 'moderator_E',
 'moderator_F',
 'moderator_FD',
 'moderator_GT',
 'moderator_IT',
 'moderator_N',
 'water_GT',
 'water_IT',
 'water_outside'}
materials = {}
my_xs = {}
xs_file = 'mgxs_6_10.h5'

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
    local x_center = (i - 0.5) * pitch - (num_cells // 2) * pitch
    local y_center = (j - 0.5) * pitch - (num_cells // 2) * pitch
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
-- for i = 1, num_cells do
--   val_table[i] = {} -- initialize inner table for each row
--   for j = 1, num_cells do
--       val_table[i][j] = -1
--       if not should_avoid(i, j) then
--           x_center, y_center = compute_cell_center(i, j, pitch)
--           my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })

            
--           xs_fuel_pincell = xs.Get(my_xs["fuel_pincell"])
--           if xs_fuel_pincell and type(xs_fuel_pincell["sigma_f"]) == "table" then
--               sig_f = xs_fuel_pincell["sigma_f"]
--           else
--               error("sigma_f is not a valid number or does not exist in the table")
--           end

--           val = 0.
--           for g = 1, num_groups do
--               ffi = fieldfunc.FFInterpolationCreate(VOLUME)
--               fieldfunc.SetProperty(ffi, OPERATION, OP_SUM)
--               fieldfunc.SetProperty(ffi, LOGICAL_VOLUME, my_lv)
--               fieldfunc.SetProperty(ffi, ADD_FIELDFUNCTION, fflist[g])
              
--               fieldfunc.Initialize(ffi)
--               fieldfunc.Execute(ffi)
--               val_g = fieldfunc.GetValue(ffi)
--               val = val + val_g * sig_f[g]
--           end
--           val_table[i][j] = val
--       end
--   end
-- end

for i = 1, num_cells do
    val_table[i] = {} -- initialize inner table for each row
    for j = 1, num_cells do
        val_table[i][j] = -1
        if not should_avoid(i, j) then
            x_center, y_center = compute_cell_center(i, j, pitch)
            my_lv = logvol.RCCLogicalVolume.Create({ r = 0.4060, x0 = x_center, y0 = y_center, z0 = -1.0, vz = 2.0 })  
            -- Boucle sur les familles pour remplacer 'fuel_pincell' par le nom de la famille
            local family_name = 'fuel_pincell'
            for family, positions in pairs(d) do
                for _, pos in ipairs(positions) do
                    if pos[1] == (i - 1) and pos[2] == (j - 1) then
                        family_name = 'fuel_' .. family
                        break
                    end
                end
                if family_name ~= 'fuel_pincell' then break end
            end

            -- Remplacement de 'fuel_pincell' par le nom de la famille correspondante
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
  ofile = io.open('power_2A_6fam.txt',"w")
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
fieldfunc.ExportToVTKMulti(fflist,'flux_2A_6')