using LowMachConvection

params = create_params(
    Nx=32, Ny=32,
    Lx=1.0, Ly=1.0,
    Ra=1e4, Pr=0.71,
    dt=1e-3,
    n_steps=5000,
    output_interval=100,
    T_hot=1.5, T_cold=0.5
)

run_simulation!(params, output_dir="output_rb")
