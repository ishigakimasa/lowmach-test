"""
    LowMachConvection

2次元低マッハ数熱対流ソルバー。
スタガード格子、Backward Euler 陰解法、PISO 圧力-速度連成を採用。
"""
module LowMachConvection

using LinearAlgebra, SparseArrays, Printf, Random

export SimParams, FlowField
export create_params, initialize_flow, run_simulation!
export apply_bc!, update_density!
export momentum_predictor!, pressure_poisson!, velocity_correction!, temperature_update!
export compute_nusselt
export save_csv, save_vtk

# ============================================================
# データ構造
# ============================================================

"""
シミュレーションパラメータを保持する不変構造体。
"""
struct SimParams
    Nx::Int; Ny::Int
    Lx::Float64; Ly::Float64
    dx::Float64; dy::Float64
    dt::Float64
    Ra::Float64; Pr::Float64
    mu::Float64; lambda::Float64; cp::Float64
    rho_ref::Float64; T_ref::Float64
    g::Float64; beta::Float64
    T_hot::Float64; T_cold::Float64
    n_steps::Int
    output_interval::Int
end

"""
流れ場変数を保持する可変構造体。スタガード格子。

- `u`   : x 方向速度, サイズ (Nx+1)×Ny,  面中央 x=(i-1)*dx, y=(j-0.5)*dy
- `v`   : y 方向速度, サイズ Nx×(Ny+1),  面中央 x=(i-0.5)*dx, y=(j-1)*dy
- `p`   : 動圧修正量, サイズ Nx×Ny,      セル中央
- `T`   : 温度,       サイズ Nx×Ny,      セル中央
- `rho` : 密度,       サイズ Nx×Ny,      セル中央
"""
mutable struct FlowField
    u::Matrix{Float64}   # (Nx+1) × Ny
    v::Matrix{Float64}   # Nx × (Ny+1)
    p::Matrix{Float64}   # Nx × Ny
    T::Matrix{Float64}   # Nx × Ny
    rho::Matrix{Float64} # Nx × Ny
end

# io.jl の関数定義（FlowField と SimParams が揃ってから読み込む）
include("io.jl")

# ============================================================
# パラメータ生成
# ============================================================

"""
    create_params(; Nx, Ny, Lx, Ly, Ra, Pr, dt, n_steps, output_interval,
                    T_hot, T_cold) → SimParams

無次元 Ra・Pr から物性値を計算して SimParams を返す。
  α = sqrt(Pr/Ra),  ν = α·Pr,  μ = ν·ρ_ref,  λ = α·ρ_ref·cp
"""
function create_params(;
        Nx::Int     = 32,
        Ny::Int     = 32,
        Lx::Float64 = 1.0,
        Ly::Float64 = 1.0,
        Ra::Float64 = 1e4,
        Pr::Float64 = 0.71,
        dt::Float64 = 1e-3,
        n_steps::Int = 5000,
        output_interval::Int = 100,
        T_hot::Float64  = 1.5,
        T_cold::Float64 = 0.5)

    dx      = Lx / Nx
    dy      = Ly / Ny
    rho_ref = 1.0
    T_ref   = 1.0
    g       = 1.0
    beta    = 1.0 / T_ref   # ideal gas
    cp      = 1.0

    alpha   = sqrt(Pr / Ra) # thermal diffusivity (dimensionless)
    nu      = alpha * Pr    # kinematic viscosity
    mu_val  = nu * rho_ref  # dynamic viscosity
    lam_val = alpha * rho_ref * cp # thermal conductivity

    return SimParams(Nx, Ny, Lx, Ly, dx, dy, dt,
                     Ra, Pr, mu_val, lam_val, cp,
                     rho_ref, T_ref, g, beta,
                     T_hot, T_cold,
                     n_steps, output_interval)
end

# ============================================================
# 初期化
# ============================================================

"""
    initialize_flow(params) → FlowField

初期条件を設定した FlowField を返す。
- 速度: ゼロ＋振幅 1e-4 のランダム擾乱
- 温度: 線形分布（下:T_hot ～ 上:T_cold）＋振幅 1e-3 のランダム擾乱
- 密度: 状態方程式から計算
"""
function initialize_flow(params::SimParams)
    Nx, Ny = params.Nx, params.Ny

    u = 1e-4 .* (rand(Nx+1, Ny)   .- 0.5)
    v = 1e-4 .* (rand(Nx,   Ny+1) .- 0.5)
    p = zeros(Float64, Nx, Ny)

    T = Matrix{Float64}(undef, Nx, Ny)
    for j in 1:Ny
        yc   = (j - 0.5) * params.dy
        Tlin = params.T_hot - (params.T_hot - params.T_cold) * yc / params.Ly
        for i in 1:Nx
            T[i,j] = Tlin + 1e-3 * (rand() - 0.5)
        end
    end

    rho = params.rho_ref .* params.T_ref ./ T

    field = FlowField(u, v, p, T, rho)
    apply_bc!(field, params)
    return field
end

# ============================================================
# 境界条件
# ============================================================

"""
    apply_bc!(field, params)

全壁面のすべりなし速度境界条件を適用する。
温度の Dirichlet / Neumann 条件は各ソルバー内の行列組立で処理する。
"""
function apply_bc!(field::FlowField, params::SimParams)
    Nx, Ny = params.Nx, params.Ny

    # 法線速度（壁面そのものに定義された面速度）を 0 に
    @. field.u[1,    :] = 0.0   # 左壁
    @. field.u[Nx+1, :] = 0.0   # 右壁
    @. field.v[:,    1] = 0.0   # 下壁
    @. field.v[:, Ny+1] = 0.0   # 上壁
    nothing
end

# ============================================================
# 状態方程式から密度更新
# ============================================================

"""
    update_density!(field, params)

理想気体近似の状態方程式 ρ = ρ_ref·T_ref / T で密度を更新する。
"""
function update_density!(field::FlowField, params::SimParams)
    @. field.rho = params.rho_ref * params.T_ref / field.T
    nothing
end

# ============================================================
# 運動量予測ステップ（Momentum Predictor）
# ============================================================

"""
    momentum_predictor!(field, field_old, params)

x・y 方向の速度を陰解法（Backward Euler）で予測する。
- 拡散項: 陰的
- 対流項・圧力勾配: 陽的（前ステップ値）
- 浮力項: -(ρ - ρ_ref)·g を y 方向 RHS に加算
"""
function momentum_predictor!(field::FlowField, field_old::FlowField,
                              params::SimParams)
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    mu = params.mu

    # ── x 方向速度 u（内部面: i=2..Nx, j=1..Ny）─────────────────────────
    nu_x = Nx - 1        # 内部面の x 方向数
    N_u  = nu_x * Ny

    # 線形インデックス: i=2..Nx, j=1..Ny → 1..N_u
    u_idx(i,j) = (j-1)*nu_x + (i-1)   # i-1 ∈ 1..nu_x

    I_u = Int[];  J_u = Int[];  V_u = Float64[]
    b_u = zeros(Float64, N_u)

    ax_u = mu / dx^2
    ay_u = mu / dy^2

    for j in 1:Ny, i in 2:Nx
        k = u_idx(i, j)

        # 面密度: u[i,j] は cell(i-1,j) と cell(i,j) の間
        rho_f = (field_old.rho[i-1,j] + field_old.rho[i,j]) * 0.5

        # ── 対角係数 ──
        diag = rho_f / dt + 2.0*ax_u
        # 上下壁のゴーストセル（すべりなし）: 対角が +ay_u 増える
        diag += (j == 1 || j == Ny) ? 2.0*ay_u : 2.0*ay_u
        # ↑ j=1 と j=Ny では d²u/dy² の係数が -3/dy² → diag に 3*ay_u
        #   それ以外は -2/dy² → diag に 2*ay_u
        # 正しく分岐する
        if j == 1 || j == Ny
            diag = rho_f / dt + 2.0*ax_u + 3.0*ay_u
        else
            diag = rho_f / dt + 2.0*ax_u + 2.0*ay_u
        end

        push!(I_u, k); push!(J_u, k); push!(V_u, diag)

        # x 方向オフ対角（壁面 u=0 はゼロ寄与なので off-diag なし）
        if i-1 >= 2
            push!(I_u, k); push!(J_u, u_idx(i-1,j)); push!(V_u, -ax_u)
        end
        if i+1 <= Nx
            push!(I_u, k); push!(J_u, u_idx(i+1,j)); push!(V_u, -ax_u)
        end

        # y 方向オフ対角
        if j > 1
            push!(I_u, k); push!(J_u, u_idx(i,j-1)); push!(V_u, -ay_u)
        end
        if j < Ny
            push!(I_u, k); push!(J_u, u_idx(i,j+1)); push!(V_u, -ay_u)
        end

        # ── RHS ──
        rhs_val = rho_f * field_old.u[i,j] / dt

        # 圧力勾配（陽的）: face(i,j) は cell(i-1,j) と cell(i,j) の間
        rhs_val -= (field_old.p[i,j] - field_old.p[i-1,j]) / dx

        # 対流項（陽的・中心差分）
        u_val = field_old.u[i,j]

        # ∂u/∂x: u[i-1,j]（左壁 0）と u[i+1,j]（右壁 0）を使う
        u_left  = field_old.u[i-1,j]   # i=2 → u[1,j]=0 already set
        u_right = field_old.u[i+1,j]   # i=Nx → u[Nx+1,j]=0 already set
        dudx = (u_right - u_left) / (2.0*dx)

        # ∂u/∂y: ゴーストセルによるすべりなし
        if j == 1
            u_above = (j < Ny) ? field_old.u[i,j+1] : -field_old.u[i,j]
            u_below = -field_old.u[i,1]           # ghost: no-slip bottom
        elseif j == Ny
            u_above = -field_old.u[i,Ny]          # ghost: no-slip top
            u_below = field_old.u[i,j-1]
        else
            u_above = field_old.u[i,j+1]
            u_below = field_old.u[i,j-1]
        end
        dudy = (u_above - u_below) / (2.0*dy)

        # v の u 面補間（4 点平均）
        v_sw = field_old.v[i-1, j  ]
        v_se = field_old.v[i,   j  ]
        v_nw = field_old.v[i-1, j+1]
        v_ne = field_old.v[i,   j+1]
        v_at_u = (v_sw + v_se + v_nw + v_ne) * 0.25

        rhs_val -= rho_f * (u_val*dudx + v_at_u*dudy)

        b_u[k] = rhs_val
    end

    A_u   = sparse(I_u, J_u, V_u, N_u, N_u)
    u_sol = A_u \ b_u

    for j in 1:Ny, i in 2:Nx
        field.u[i,j] = u_sol[u_idx(i,j)]
    end

    # ── y 方向速度 v（内部面: i=1..Nx, j=2..Ny）─────────────────────────
    nv_x = Nx
    N_v  = nv_x * (Ny-1)

    # 線形インデックス: i=1..Nx, j=2..Ny → 1..N_v
    v_idx(i,j) = (j-2)*nv_x + i

    I_v = Int[];  J_v = Int[];  V_v = Float64[]
    b_v = zeros(Float64, N_v)

    ax_v = mu / dx^2
    ay_v = mu / dy^2

    for j in 2:Ny, i in 1:Nx
        k = v_idx(i, j)

        # 面密度: v[i,j] は cell(i,j-1) と cell(i,j) の間
        rho_f = (field_old.rho[i,j-1] + field_old.rho[i,j]) * 0.5

        # ── 対角係数 ──
        if i == 1 || i == Nx
            diag = rho_f / dt + 3.0*ax_v + 2.0*ay_v
        else
            diag = rho_f / dt + 2.0*ax_v + 2.0*ay_v
        end

        push!(I_v, k); push!(J_v, k); push!(V_v, diag)

        # y 方向オフ対角（壁面 v=0 はゼロ寄与）
        if j-1 >= 2
            push!(I_v, k); push!(J_v, v_idx(i,j-1)); push!(V_v, -ay_v)
        end
        if j+1 <= Ny
            push!(I_v, k); push!(J_v, v_idx(i,j+1)); push!(V_v, -ay_v)
        end

        # x 方向オフ対角（ゴーストセルによるすべりなし）
        if i > 1
            push!(I_v, k); push!(J_v, v_idx(i-1,j)); push!(V_v, -ax_v)
        end
        if i < Nx
            push!(I_v, k); push!(J_v, v_idx(i+1,j)); push!(V_v, -ax_v)
        end

        # ── RHS ──
        rhs_val = rho_f * field_old.v[i,j] / dt

        # 圧力勾配（陽的）
        rhs_val -= (field_old.p[i,j] - field_old.p[i,j-1]) / dy

        # 浮力: -(ρ - ρ_ref)·g（重力は -y 方向）
        rhs_val -= (rho_f - params.rho_ref) * params.g

        # 対流項（陽的・中心差分）
        v_val = field_old.v[i,j]

        # ∂v/∂x: ゴーストセルによるすべりなし
        if i == 1
            v_right = (i < Nx) ? field_old.v[i+1,j] : -field_old.v[i,j]
            v_left  = -field_old.v[1,j]           # ghost: no-slip left
        elseif i == Nx
            v_right = -field_old.v[Nx,j]          # ghost: no-slip right
            v_left  = field_old.v[i-1,j]
        else
            v_right = field_old.v[i+1,j]
            v_left  = field_old.v[i-1,j]
        end
        dvdx = (v_right - v_left) / (2.0*dx)

        # ∂v/∂y: v[i,1]=0, v[i,Ny+1]=0 は配列に直接保持
        v_above = field_old.v[i, j+1]   # j+1 ≤ Ny+1, wall value=0
        v_below = field_old.v[i, j-1]   # j-1 ≥ 1,    wall value=0
        dvdy = (v_above - v_below) / (2.0*dy)

        # u の v 面補間（4 点平均）
        u_sw = field_old.u[i,   j-1]
        u_se = field_old.u[i+1, j-1]
        u_nw = field_old.u[i,   j  ]
        u_ne = field_old.u[i+1, j  ]
        u_at_v = (u_sw + u_se + u_nw + u_ne) * 0.25

        rhs_val -= rho_f * (u_at_v*dvdx + v_val*dvdy)

        b_v[k] = rhs_val
    end

    A_v   = sparse(I_v, J_v, V_v, N_v, N_v)
    v_sol = A_v \ b_v

    for j in 2:Ny, i in 1:Nx
        field.v[i,j] = v_sol[v_idx(i,j)]
    end

    nothing
end

# ============================================================
# 圧力ポアソン方程式
# ============================================================

"""
    pressure_poisson!(field, params) → dp::Matrix{Float64}

速度場の発散から圧力修正量 dp を求める。

    ∇²dp = (1/Δt) · ∇·u*

全壁面に Neumann 条件（∂dp/∂n = 0）を適用。
セル (1,1) を dp=0 に固定して行列を正則化する。
"""
function pressure_poisson!(field::FlowField, params::SimParams)
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    N = Nx * Ny

    p_idx(i,j) = (j-1)*Nx + i

    I_p = Int[];  J_p = Int[];  V_p = Float64[]
    b_p = zeros(Float64, N)

    ax = 1.0 / dx^2
    ay = 1.0 / dy^2

    for j in 1:Ny, i in 1:Nx
        k = p_idx(i, j)

        if i == 1 && j == 1
            # セル (1,1) を Dirichlet 固定: dp = 0
            push!(I_p, k); push!(J_p, k); push!(V_p, 1.0)
            b_p[k] = 0.0
            continue
        end

        # RHS: 速度発散 / dt
        div_val = (field.u[i+1,j] - field.u[i,j]) / dx +
                  (field.v[i,j+1] - field.v[i,j]) / dy
        b_p[k] = div_val / dt

        # ─ x 方向ラプラシアン（Neumann BC）─
        diag_x = 0.0
        if i == 1
            diag_x -= ax
            push!(I_p, k); push!(J_p, p_idx(2,j));    push!(V_p,  ax)
        elseif i == Nx
            diag_x -= ax
            push!(I_p, k); push!(J_p, p_idx(Nx-1,j)); push!(V_p,  ax)
        else
            diag_x -= 2.0*ax
            push!(I_p, k); push!(J_p, p_idx(i-1,j));  push!(V_p,  ax)
            push!(I_p, k); push!(J_p, p_idx(i+1,j));  push!(V_p,  ax)
        end

        # ─ y 方向ラプラシアン（Neumann BC）─
        diag_y = 0.0
        if j == 1
            diag_y -= ay
            push!(I_p, k); push!(J_p, p_idx(i,2));    push!(V_p,  ay)
        elseif j == Ny
            diag_y -= ay
            push!(I_p, k); push!(J_p, p_idx(i,Ny-1)); push!(V_p,  ay)
        else
            diag_y -= 2.0*ay
            push!(I_p, k); push!(J_p, p_idx(i,j-1));  push!(V_p,  ay)
            push!(I_p, k); push!(J_p, p_idx(i,j+1));  push!(V_p,  ay)
        end

        push!(I_p, k); push!(J_p, k); push!(V_p, diag_x + diag_y)
    end

    A_p    = sparse(I_p, J_p, V_p, N, N)
    dp_vec = A_p \ b_p

    return reshape(dp_vec, Nx, Ny)
end

# ============================================================
# 速度修正（Velocity Correction）
# ============================================================

"""
    velocity_correction!(field, dp, params)

圧力修正量 dp から速度を修正する。

    u** = u* - Δt · (dp[i,j] - dp[i-1,j]) / (Δx · ρ̄_face)
    v** = v* - Δt · (dp[i,j] - dp[i,j-1]) / (Δy · ρ̄_face)
"""
function velocity_correction!(field::FlowField, dp::Matrix{Float64},
                               params::SimParams)
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt

    # u 内部面 (i=2..Nx)
    for j in 1:Ny, i in 2:Nx
        rho_f = (field.rho[i-1,j] + field.rho[i,j]) * 0.5
        field.u[i,j] -= dt * (dp[i,j] - dp[i-1,j]) / (dx * rho_f)
    end

    # v 内部面 (j=2..Ny)
    for j in 2:Ny, i in 1:Nx
        rho_f = (field.rho[i,j-1] + field.rho[i,j]) * 0.5
        field.v[i,j] -= dt * (dp[i,j] - dp[i,j-1]) / (dy * rho_f)
    end

    nothing
end

# ============================================================
# 温度更新（陰解法）
# ============================================================

"""
    temperature_update!(field, field_old, params)

エネルギー方程式を Backward Euler で解く。
- 拡散項: 陰的
- 対流項: 陽的
- 境界条件: 下壁 T_hot・上壁 T_cold（Dirichlet）、左右壁断熱（Neumann）
"""
function temperature_update!(field::FlowField, field_old::FlowField,
                              params::SimParams)
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    lam = params.lambda
    cp  = params.cp
    T_hot  = params.T_hot
    T_cold = params.T_cold

    N = Nx * Ny
    t_idx(i,j) = (j-1)*Nx + i

    I_t = Int[];  J_t = Int[];  V_t = Float64[]
    b_t = zeros(Float64, N)

    for j in 1:Ny, i in 1:Nx
        k     = t_idx(i, j)
        rho_c = field_old.rho[i,j]
        diag  = rho_c * cp / dt

        # ─ x 方向拡散（Neumann BC: 断熱）─
        ax = lam / dx^2
        if i == 1
            # ghost T[0,j] = T[1,j] → 対角 +ax のみ
            diag += ax
            push!(I_t, k); push!(J_t, t_idx(2,j));    push!(V_t, -ax)
        elseif i == Nx
            diag += ax
            push!(I_t, k); push!(J_t, t_idx(Nx-1,j)); push!(V_t, -ax)
        else
            diag += 2.0*ax
            push!(I_t, k); push!(J_t, t_idx(i-1,j)); push!(V_t, -ax)
            push!(I_t, k); push!(J_t, t_idx(i+1,j)); push!(V_t, -ax)
        end

        # ─ y 方向拡散（Dirichlet BC: T_hot / T_cold）─
        ay = lam / dy^2
        if j == 1
            # ghost T[i,0] = 2*T_hot - T[i,1]
            # → d²T/dy²|_{j=1} = (T[i,2] - 3T[i,1]) / dy² + 2T_hot/dy²
            diag  += 3.0*ay
            push!(I_t, k); push!(J_t, t_idx(i,2));    push!(V_t, -ay)
            b_t[k] += 2.0 * ay * T_hot
        elseif j == Ny
            # ghost T[i,Ny+1] = 2*T_cold - T[i,Ny]
            diag  += 3.0*ay
            push!(I_t, k); push!(J_t, t_idx(i,Ny-1)); push!(V_t, -ay)
            b_t[k] += 2.0 * ay * T_cold
        else
            diag += 2.0*ay
            push!(I_t, k); push!(J_t, t_idx(i,j-1)); push!(V_t, -ay)
            push!(I_t, k); push!(J_t, t_idx(i,j+1)); push!(V_t, -ay)
        end

        push!(I_t, k); push!(J_t, k); push!(V_t, diag)

        # ─ RHS: ρ·cp·T_old / Δt ─
        b_t[k] += rho_c * cp * field_old.T[i,j] / dt

        # ─ 対流項（陽的・中心差分）─
        u_c = (field_old.u[i,j] + field_old.u[i+1,j]) * 0.5
        v_c = (field_old.v[i,j] + field_old.v[i,j+1]) * 0.5

        # ∂T/∂x
        if i == 1
            dTdx = (field_old.T[2,j]   - field_old.T[1,j])   / dx
        elseif i == Nx
            dTdx = (field_old.T[Nx,j]  - field_old.T[Nx-1,j]) / dx
        else
            dTdx = (field_old.T[i+1,j] - field_old.T[i-1,j]) / (2.0*dx)
        end

        # ∂T/∂y（境界ではゴーストセル）
        if j == 1
            T_ghost_bot = 2.0*T_hot - field_old.T[i,1]
            dTdy = (field_old.T[i,2] - T_ghost_bot) / (2.0*dy)
        elseif j == Ny
            T_ghost_top = 2.0*T_cold - field_old.T[i,Ny]
            dTdy = (T_ghost_top - field_old.T[i,Ny-1]) / (2.0*dy)
        else
            dTdy = (field_old.T[i,j+1] - field_old.T[i,j-1]) / (2.0*dy)
        end

        b_t[k] -= rho_c * cp * (u_c*dTdx + v_c*dTdy)
    end

    A_t   = sparse(I_t, J_t, V_t, N, N)
    T_vec = A_t \ b_t

    field.T .= reshape(T_vec, Nx, Ny)
    nothing
end

# ============================================================
# Nusselt 数計算
# ============================================================

"""
    compute_nusselt(field, params) → Float64

下壁（y=0）での平均熱流束から Nusselt 数を計算する。

    Nu = (Ly / ΔT) · (1/Nx) · Σᵢ [(T_hot - T[i,1]) / (dy/2)]
"""
function compute_nusselt(field::FlowField, params::SimParams)
    Nx = params.Nx
    dy = params.dy
    ΔT = params.T_hot - params.T_cold
    Ly = params.Ly

    flux = 0.0
    for i in 1:Nx
        # ∂T/∂y|_{wall} ≈ (T[i,1] - T_hot) / (dy/2)  →  heat flux = -λ·∂T/∂y
        flux += (params.T_hot - field.T[i,1]) / (dy * 0.5)
    end
    avg_flux = flux / Nx    # average ∂T/∂y magnitude at bottom wall

    Nu = (Ly / ΔT) * avg_flux
    return Nu
end

# ============================================================
# メインシミュレーションループ
# ============================================================

"""
    run_simulation!(params; output_dir="output") → FlowField

PISO 法による時間発展シミュレーションを実行する。

各タイムステップ:
1. Momentum Predictor（陰解法で中間速度 u*, v* を予測）
2. Pressure Corrector 1（ポアソン方程式 → 速度・圧力修正）
3. Pressure Corrector 2（2 回目の圧力・速度修正）
4. 温度輸送方程式を陰解法で解く
5. 密度を状態方程式で更新
"""
function run_simulation!(params::SimParams; output_dir::String="output")
    mkpath(output_dir)

    field = initialize_flow(params)

    # 初期出力
    save_csv(field, params, 0, output_dir)
    save_vtk(field, params, 0, output_dir)

    for step in 1:params.n_steps
        field_old = deepcopy(field)

        # ── 1. Momentum Predictor ──────────────────────────────────────────
        momentum_predictor!(field, field_old, params)
        apply_bc!(field, params)

        # ── 2. Pressure Corrector 1 ────────────────────────────────────────
        dp1 = pressure_poisson!(field, params)
        field.p .+= dp1
        velocity_correction!(field, dp1, params)
        apply_bc!(field, params)

        # ── 3. Pressure Corrector 2 ────────────────────────────────────────
        dp2 = pressure_poisson!(field, params)
        field.p .+= dp2
        velocity_correction!(field, dp2, params)
        apply_bc!(field, params)

        # ── 4. Temperature Update ──────────────────────────────────────────
        temperature_update!(field, field_old, params)
        apply_bc!(field, params)

        # ── 5. Density Update ─────────────────────────────────────────────
        update_density!(field, params)

        # ── 6. モニタリング ────────────────────────────────────────────────
        Nu = compute_nusselt(field, params)
        @printf("step=%5d, Nu=%8.4f\n", step, Nu)

        # ── 7. 出力 ───────────────────────────────────────────────────────
        if step % params.output_interval == 0
            save_csv(field, params, step, output_dir)
            save_vtk(field, params, step, output_dir)
        end
    end

    return field
end

end # module LowMachConvection
