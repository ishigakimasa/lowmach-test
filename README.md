# LowMachConvection.jl

2次元低マッハ数熱対流解析コード（Julia）

## 概要

このパッケージは **低マッハ数近似（Low Mach number approximation）** に基づく
2次元熱対流の数値解析ソルバーです。
スタガード格子、Backward Euler 陰解法、PISO 圧力-速度連成アルゴリズムを組み合わせ、
密度変化を考慮しながら音波を除去した効率的な計算を実現しています。
テスト問題として **Rayleigh-Bénard 対流** を扱います。

---

## 物理モデル

### 低マッハ数近似

圧力を熱力学的圧力 $p_0(t)$ と動圧 $p'(\mathbf{x},t)$ に分離します。
閉じた系では $p_0$ は一定とみなし、状態方程式は

$$\rho T = \rho_0 T_0 = \text{const} \quad \Longrightarrow \quad \rho = \frac{\rho_\mathrm{ref} T_\mathrm{ref}}{T}$$

### 支配方程式

| 方程式 | 式 |
|--------|-----|
| 連続の式 | $\dfrac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$ |
| 運動量 | $\rho \dfrac{D\mathbf{u}}{Dt} = -\nabla p' + \nabla \cdot (\mu \nabla \mathbf{u}) + (\rho-\rho_\mathrm{ref})\mathbf{g}$ |
| エネルギー | $\rho c_p \dfrac{DT}{Dt} = \nabla \cdot (\lambda \nabla T)$ |

重力は $-y$ 方向で大きさ $g=1$（無次元）、浮力項は低マッハ数 Boussinesq 的に処理します。

---

## 数値手法

### スタガード格子

```
  v[i,j+1]  ←  y 上面中央
     ↑
-----+-----     ← p, T, ρ はセル中央
     |
u[i,j] -- cell(i,j) -- u[i+1,j]   (x 面中央)
     |
-----+-----
  v[i,j]    ←  y 下面中央
```

| 変数 | 配置 | サイズ |
|------|------|--------|
| `u`  | x 面中央 | $(N_x+1)\times N_y$ |
| `v`  | y 面中央 | $N_x\times(N_y+1)$ |
| `p, T, ρ` | セル中央 | $N_x\times N_y$ |

### 時間積分

**Backward Euler（完全陰解法）**。各ステップで疎行列を構築し `\` 演算子（Julia の直接ソルバー）で解きます。

### 圧力-速度連成（PISO法）

1. **Momentum Predictor**: 陰的拡散・陽的対流・陽的圧力勾配で中間速度 $u^*, v^*$ を予測
2. **Pressure Corrector 1**: ポアソン方程式 $\nabla^2 p' = \frac{1}{\Delta t}\nabla\cdot\mathbf{u}^*$ を解き、速度・圧力を修正
3. **Pressure Corrector 2**: 2 回目の圧力修正
4. **Temperature Update**: エネルギー方程式を陰解法で解く
5. **Density Update**: 状態方程式から密度を更新

---

## テスト問題：Rayleigh-Bénard 対流

| 項目 | 値 |
|------|----|
| 計算領域 | $[0,1]\times[0,1]$ |
| 格子 | $32\times32$ |
| Ra | $1\times10^4$ |
| Pr | $0.71$ |
| 下壁温度 $T_\mathrm{hot}$ | 1.5 |
| 上壁温度 $T_\mathrm{cold}$ | 0.5 |
| 時間刻み $\Delta t$ | $10^{-3}$ |
| ステップ数 | 5000 |

### 境界条件

| 壁面 | 速度 | 温度 |
|------|------|------|
| 下壁 ($y=0$) | すべりなし | $T=T_\mathrm{hot}$（等温） |
| 上壁 ($y=L_y$) | すべりなし | $T=T_\mathrm{cold}$（等温） |
| 左右壁 | すべりなし | 断熱 ($\partial T/\partial x=0$) |
| 圧力 | — | Neumann ($\partial p'/\partial n=0$) |

---

## インストール・実行方法

### 必要環境

- Julia 1.6 以上
- 標準ライブラリのみ使用（`LinearAlgebra`, `SparseArrays`, `Printf`, `Random`）

### 実行手順

```bash
# リポジトリのルートで
julia test/rayleigh_benard.jl
```

出力ファイルは `output_rb/` ディレクトリに保存されます。

---

## パラメータ説明

`create_params` のキーワード引数：

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `Nx`, `Ny` | 32, 32 | 格子分割数 |
| `Lx`, `Ly` | 1.0, 1.0 | 計算領域の大きさ |
| `Ra` | 1e4 | Rayleigh 数 |
| `Pr` | 0.71 | Prandtl 数（空気） |
| `dt` | 1e-3 | 時間刻み幅 |
| `n_steps` | 5000 | 総ステップ数 |
| `output_interval` | 100 | 出力間隔（ステップ） |
| `T_hot` | 1.5 | 下壁温度 |
| `T_cold` | 0.5 | 上壁温度 |

物性値は Ra・Pr から自動計算：

$$\alpha = \sqrt{Pr/Ra},\quad \nu = \alpha Pr,\quad \mu = \nu\rho_\mathrm{ref},\quad \lambda = \alpha\rho_\mathrm{ref}c_p$$

---

## 出力ファイル

`output_dir/` に以下のファイルが保存されます：

| ファイル | 内容 |
|---------|------|
| `T_XXXXXX.csv` | 温度場（x, y, T） |
| `u_XXXXXX.csv` | x 速度（セル中央補間） |
| `v_XXXXXX.csv` | y 速度（セル中央補間） |
| `p_XXXXXX.csv` | 動圧修正量 |
| `rho_XXXXXX.csv` | 密度場 |
| `flow_XXXXXX.vtk` | VTK ASCII legacy 形式（ParaView 等で可視化可能） |

---

## ファイル構成

```
LowMachConvection/
├── Project.toml              # パッケージ設定
├── README.md                 # このファイル
├── src/
│   ├── LowMachConvection.jl  # メインモジュール
│   └── io.jl                 # CSV/VTK 出力
└── test/
    └── rayleigh_benard.jl    # Rayleigh-Bénard 対流テスト
```
