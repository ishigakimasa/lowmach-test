# io.jl – output routines for LowMachConvection

"""
    save_csv(field, params, step, dir)

Save T, u_cell, v_cell, p, rho at cell centres to CSV files inside `dir`.
Velocities are interpolated from staggered faces to cell centres before writing.
"""
function save_csv(field::FlowField, params::SimParams, step::Int, dir::String)
    Nx, Ny = params.Nx, params.Ny
    dx, dy = params.dx, params.dy

    # ── interpolate u to cell centres ──────────────────────────────────────
    u_c = 0.5 .* (field.u[1:Nx, :] .+ field.u[2:Nx+1, :])   # Nx×Ny
    # ── interpolate v to cell centres ──────────────────────────────────────
    v_c = 0.5 .* (field.v[:, 1:Ny] .+ field.v[:, 2:Ny+1])   # Nx×Ny

    function write_field(fname, data)
        open(fname, "w") do io
            for j in 1:Ny
                yc = (j - 0.5) * dy
                for i in 1:Nx
                    xc = (i - 0.5) * dx
                    println(io, "$xc,$yc,$(data[i,j])")
                end
            end
        end
    end

    write_field(joinpath(dir, @sprintf("T_%06d.csv",   step)), field.T)
    write_field(joinpath(dir, @sprintf("u_%06d.csv",   step)), u_c)
    write_field(joinpath(dir, @sprintf("v_%06d.csv",   step)), v_c)
    write_field(joinpath(dir, @sprintf("p_%06d.csv",   step)), field.p)
    write_field(joinpath(dir, @sprintf("rho_%06d.csv", step)), field.rho)
end

"""
    save_vtk(field, params, step, dir)

Save all fields to a VTK ASCII legacy file (structured grid) inside `dir`.
Cell-centred scalars T, p, rho and vector (u,v) are written.
"""
function save_vtk(field::FlowField, params::SimParams, step::Int, dir::String)
    Nx, Ny = params.Nx, params.Ny
    dx, dy = params.dx, params.dy

    u_c = 0.5 .* (field.u[1:Nx, :] .+ field.u[2:Nx+1, :])
    v_c = 0.5 .* (field.v[:, 1:Ny] .+ field.v[:, 2:Ny+1])

    fname = joinpath(dir, @sprintf("flow_%06d.vtk", step))
    open(fname, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "LowMach step $step")
        println(io, "ASCII")
        println(io, "DATASET RECTILINEAR_GRID")
        println(io, "DIMENSIONS $(Nx+1) $(Ny+1) 1")

        print(io, "X_COORDINATES $(Nx+1) float\n")
        for i in 0:Nx; print(io, "$(i*dx) "); end; println(io)

        print(io, "Y_COORDINATES $(Ny+1) float\n")
        for j in 0:Ny; print(io, "$(j*dy) "); end; println(io)

        println(io, "Z_COORDINATES 1 float\n0.0")

        N = Nx * Ny
        println(io, "CELL_DATA $N")

        # Temperature
        println(io, "SCALARS Temperature float 1")
        println(io, "LOOKUP_TABLE default")
        for j in 1:Ny, i in 1:Nx; println(io, field.T[i,j]); end

        # Pressure
        println(io, "SCALARS Pressure float 1")
        println(io, "LOOKUP_TABLE default")
        for j in 1:Ny, i in 1:Nx; println(io, field.p[i,j]); end

        # Density
        println(io, "SCALARS Density float 1")
        println(io, "LOOKUP_TABLE default")
        for j in 1:Ny, i in 1:Nx; println(io, field.rho[i,j]); end

        # Velocity vector
        println(io, "VECTORS Velocity float")
        for j in 1:Ny, i in 1:Nx
            println(io, "$(u_c[i,j]) $(v_c[i,j]) 0.0")
        end
    end
end
