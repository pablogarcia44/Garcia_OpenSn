--- Import mesh

my_filename = "mesh_1fuel_fine.obj"

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
-- mesh.ExportToPVTU("uox_1fuel")
-- mesh.ComputeVolumePerMaterialID()
-- os.exit()

-- Create cross sections
my_xs = {}
xs_file = 'mgxs.h5'


my_xs["moderator"] = xs.Create()
xs.Set(my_xs["moderator"],OPENMC_XSLIB,xs_file,565.0,"moderator")
my_xs["clad"] = xs.Create()
xs.Set(my_xs["clad"],OPENMC_XSLIB,xs_file,565.0,"clad")
my_xs["fuel"] = xs.Create()
xs.Set(my_xs["fuel"],OPENMC_XSLIB,xs_file,565.0,"fuel")




-- xs.SetScalingFactor(my_xs["moderator"], 1)
-- xs.SetScalingFactor(my_xs["clad"], 1)
-- xs.SetScalingFactor(my_xs["fuel"], 1)


-- Create materials

materials = {}


materials[1] = mat.AddMaterial("clad")
mat.SetProperty(materials[1], TRANSPORT_XSECTIONS, EXISTING, my_xs["clad"])
materials[2] = mat.AddMaterial("fuel")
mat.SetProperty(materials[2], TRANSPORT_XSECTIONS, EXISTING, my_xs["fuel"])
materials[3] = mat.AddMaterial("moderator")
mat.SetProperty(materials[3], TRANSPORT_XSECTIONS, EXISTING, my_xs["moderator"])




-- Setup Physics

-- Angular quadrature
pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV,32,4)
aquad.OptimizeForPolarSymmetry(pquad, 4.0*math.pi)

-- Solver
num_groups = 172

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
    power_field_function_on = true,
    power_default_kappa = 1.0,
    power_normalization = 1.0,
    save_angular_flux = true,
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

k_solver = lbs.PowerIterationKEigen.Create({
  lbs_solver_handle = phys1,
  k_tol = 1.0e-8,
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


fflist,count = lbs.GetScalarFieldFunctionList(phys1)
fieldfunc.ExportToVTKMulti(fflist,'pin_1fuel')
