--- Import mesh

my_filename = "lattice_1.obj"

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
mesh.ExportToPVTU("mesh_1A")
-- mesh.ComputeVolumePerMaterialID()
-- os.exit()

-- Create cross sections


mat_names = {'clad', 'fuel', 'gap', 'moderator'}
materials = {}
my_xs = {}
xs_file = 'mgxs_1A.h5'

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
MPIBarrier()
fieldfunc.ExportToVTKMulti(fflist,'flux_1A')

