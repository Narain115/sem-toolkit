import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.stats import f as f_dist
import itertools
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ── WHAT THIS MODULE DOES ─────────────────────────────────────────────────────
# Simulates a Design of Experiments (DOE) for a PECVD silicon nitride process
# PECVD (Plasma Enhanced CVD) deposits Si3N4 films used as passivation layers
# Process engineers run DOEs to find which parameters control film quality
#
# We model 4 factors:
#   A: Temperature (250-350°C)
#   B: RF Power (50-150W)
#   C: Pressure (1-3 Torr)
#   D: SiH4 Flow (20-60 sccm)
#
# Responses:
#   - Deposition rate (nm/min): want high
#   - Film stress (MPa): want low
#   - Refractive index: want 1.85-2.05 (quality indicator)
#   - Uniformity (%): want low (tight = good)

print("Running PECVD Si3N4 Process DOE Simulation...")

# ── 1. DEFINE PROCESS MODEL ───────────────────────────────────────────────────
# These equations are based on published PECVD process relationships
# Each response is a physics-informed function of the four factors

def pecvd_model(T, P_rf, P_gas, Q_sih4):
    """
    Simulate PECVD Si3N4 process responses.
    
    T:      Temperature (°C), range 250-350
    P_rf:   RF Power (W), range 50-150  
    P_gas:  Pressure (Torr), range 1-3
    Q_sih4: SiH4 flow (sccm), range 20-60
    
    Returns: dep_rate, stress, n_index, uniformity
    """
    # Normalize inputs to -1 to +1 (coded variables)
    T_c      = (T - 300) / 50
    Prf_c    = (P_rf - 100) / 50
    Pgas_c   = (P_gas - 2) / 1
    Qsih4_c  = (Q_sih4 - 40) / 20
    
    # Deposition rate (nm/min): driven by RF power and SiH4 flow
    dep_rate = (
        45
        + 12 * Prf_c
        + 8  * Qsih4_c
        + 4  * T_c
        - 3  * Pgas_c
        + 3  * Prf_c * Qsih4_c
        - 2  * T_c * Pgas_c
        + np.random.normal(0, 1.5)
    )
    
    # Film stress (MPa): tensile positive, compressive negative
    # Higher temp = more tensile, higher RF = more compressive
    stress = (
        -50
        + 80  * T_c
        - 60  * Prf_c
        + 20  * Pgas_c
        - 15  * Qsih4_c
        + 25  * T_c * Prf_c
        + np.random.normal(0, 5)
    )
    
    # Refractive index: quality indicator for film stoichiometry
    # Target: 1.85-2.05 (Si-rich = high n, N-rich = low n)
    n_index = (
        1.92
        + 0.08 * Prf_c
        - 0.05 * T_c
        + 0.03 * Qsih4_c
        - 0.02 * Pgas_c
        + 0.02 * Prf_c * T_c
        + np.random.normal(0, 0.01)
    )
    
    # Uniformity (1-sigma %): lower is better
    uniformity = (
        2.5
        + 0.8  * abs(Pgas_c)
        - 0.5  * T_c
        + 0.4  * abs(Qsih4_c)
        + 0.3  * Prf_c**2
        + np.random.normal(0, 0.15)
    )
    uniformity = max(0.5, uniformity)
    
    return dep_rate, stress, n_index, uniformity

# ── 2. BUILD FULL FACTORIAL DOE ───────────────────────────────────────────────
# 2^4 full factorial: each factor at 2 levels (low=-1, high=+1)
# 16 runs + 4 center points = 20 total runs
# Center points check for curvature in the response

print("Building 2^4 full factorial design...")

np.random.seed(42)

# Factor levels
levels = {
    'Temperature':  [250, 350],
    'RF_Power':     [50, 150],
    'Pressure':     [1, 3],
    'SiH4_Flow':    [20, 60]
}

# Generate all 16 factorial combinations
factorial_runs = list(itertools.product([0, 1], repeat=4))
center_runs = [(0.5, 0.5, 0.5, 0.5)] * 4  # 4 center points

all_runs = factorial_runs + center_runs

# Convert to actual values and run model
results = []
for run in all_runs:
    T_val  = levels['Temperature'][int(round(run[0]))] if run[0] in [0,1] else 300
    P_val  = levels['RF_Power'][int(round(run[1]))]    if run[1] in [0,1] else 100
    Pg_val = levels['Pressure'][int(round(run[2]))]    if run[2] in [0,1] else 2
    Q_val  = levels['SiH4_Flow'][int(round(run[3]))]   if run[3] in [0,1] else 40
    
    dep, stress, n_idx, unif = pecvd_model(T_val, P_val, Pg_val, Q_val)
    results.append({
        'Temperature': T_val, 'RF_Power': P_val,
        'Pressure': Pg_val, 'SiH4_Flow': Q_val,
        'Dep_Rate': round(dep, 2), 'Stress': round(stress, 1),
        'Ref_Index': round(n_idx, 4), 'Uniformity': round(unif, 3)
    })

doe_df = pd.DataFrame(results)
doe_df.to_csv('outputs/module4_doe_results.csv', index=False)
print(f"DOE runs: {len(doe_df)}")
print(doe_df[['Temperature','RF_Power','Dep_Rate','Stress','Ref_Index','Uniformity']].to_string(index=False))

# ── 3. MAIN EFFECTS PLOT ──────────────────────────────────────────────────────
print("\nGenerating main effects plots...")

factorial_df = doe_df.iloc[:16].copy()
factors = ['Temperature', 'RF_Power', 'Pressure', 'SiH4_Flow']
responses = ['Dep_Rate', 'Stress', 'Ref_Index', 'Uniformity']
response_labels = ['Dep. Rate (nm/min)', 'Stress (MPa)', 
                   'Ref. Index', 'Uniformity (1σ%)']
response_targets = ['Maximize', 'Minimize |value|', 
                    'Target: 1.85-2.05', 'Minimize']

fig, axes = plt.subplots(4, 4, figsize=(16, 12))

for row, (resp, label, target) in enumerate(
        zip(responses, response_labels, response_targets)):
    for col, factor in enumerate(factors):
        ax = axes[row, col]
        
        low_mean  = factorial_df[factorial_df[factor] == 
                                 levels[factor][0]][resp].mean()
        high_mean = factorial_df[factorial_df[factor] == 
                                 levels[factor][1]][resp].mean()
        
        ax.plot([levels[factor][0], levels[factor][1]], 
                [low_mean, high_mean], 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel(factor.replace('_', ' '), fontsize=8)
        
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
        if row == 0:
            ax.set_title(factor.replace('_', ' '), fontsize=9, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

plt.suptitle('PECVD Si₃N₄ Process DOE — Main Effects Plot\n'
             'Effect of Each Factor on All Four Responses',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module4_main_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/module4_main_effects.png")

# ── 4. RESPONSE SURFACE ───────────────────────────────────────────────────────
# Response surface shows how TWO factors jointly affect a response
# This is how process engineers find the optimal operating window

print("Generating response surface plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), 
                          subplot_kw={'projection': '3d'})

# Surface 1: Dep Rate vs Temperature and RF Power
T_range  = np.linspace(250, 350, 30)
P_range  = np.linspace(50, 150, 30)
T_grid, P_grid = np.meshgrid(T_range, P_range)

dep_grid = np.zeros_like(T_grid)
for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        d, _, _, _ = pecvd_model(T_grid[i,j], P_grid[i,j], 2, 40)
        dep_grid[i,j] = d

surf1 = axes[0].plot_surface(T_grid, P_grid, dep_grid, 
                              cmap='viridis', alpha=0.85)
axes[0].set_xlabel('Temperature (°C)', fontsize=9, labelpad=8)
axes[0].set_ylabel('RF Power (W)', fontsize=9, labelpad=8)
axes[0].set_zlabel('Dep. Rate (nm/min)', fontsize=9, labelpad=8)
axes[0].set_title('Deposition Rate\nvs Temperature & RF Power', 
                   fontsize=10, fontweight='bold')
fig.colorbar(surf1, ax=axes[0], shrink=0.5, label='nm/min')

# Surface 2: Film Stress vs Temperature and RF Power
stress_grid = np.zeros_like(T_grid)
for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        _, s, _, _ = pecvd_model(T_grid[i,j], P_grid[i,j], 2, 40)
        stress_grid[i,j] = s

surf2 = axes[1].plot_surface(T_grid, P_grid, stress_grid, 
                              cmap='RdYlGn_r', alpha=0.85)
axes[1].set_xlabel('Temperature (°C)', fontsize=9, labelpad=8)
axes[1].set_ylabel('RF Power (W)', fontsize=9, labelpad=8)
axes[1].set_zlabel('Stress (MPa)', fontsize=9, labelpad=8)
axes[1].set_title('Film Stress\nvs Temperature & RF Power',
                   fontsize=10, fontweight='bold')
fig.colorbar(surf2, ax=axes[1], shrink=0.5, label='MPa')

plt.suptitle('PECVD Si₃N₄ — Response Surface Analysis\n'
             'Pressure=2 Torr, SiH4 Flow=40 sccm (held constant)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module4_response_surface.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/module4_response_surface.png")

# ── 5. PROCESS OPTIMIZATION ───────────────────────────────────────────────────
# Find the optimal process conditions using multi-objective optimization
# This is the engineering output: recommended recipe settings

print("Running process optimization...")

def objective_function(params):
    T, P_rf, P_gas, Q = params
    dep, stress, n_idx, unif = pecvd_model(T, P_rf, P_gas, Q)
    
    # Desirability scores (0-1, higher is better)
    # Dep rate: target >50 nm/min
    d_dep = min(1.0, max(0, (dep - 30) / 30))
    
    # Stress: target -100 to +100 MPa
    d_stress = max(0, 1 - abs(stress) / 200)
    
    # Refractive index: target 1.90-2.00
    d_nidx = max(0, 1 - abs(n_idx - 1.95) / 0.15)
    
    # Uniformity: target <2%
    d_unif = max(0, 1 - unif / 4)
    
    # Combined desirability (geometric mean)
    combined = (d_dep * d_stress * d_nidx * d_unif) ** 0.25
    return -combined  # minimize negative = maximize desirability

# Bounds: [T, RF_Power, Pressure, SiH4_Flow]
bounds = [(250, 350), (50, 150), (1, 3), (20, 60)]
best_result = None
best_score = np.inf

# Run optimization from multiple starting points
for _ in range(20):
    x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
    result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds)
    if result.fun < best_score:
        best_score = result.fun
        best_result = result

T_opt, P_opt, Pg_opt, Q_opt = best_result.x
dep_opt, stress_opt, n_opt, unif_opt = pecvd_model(T_opt, P_opt, Pg_opt, Q_opt)

print(f"\n── Optimal PECVD Recipe ──")
print(f"  Temperature:  {T_opt:.1f} °C")
print(f"  RF Power:     {P_opt:.1f} W")
print(f"  Pressure:     {Pg_opt:.2f} Torr")
print(f"  SiH4 Flow:    {Q_opt:.1f} sccm")
print(f"\n── Predicted Responses ──")
print(f"  Dep. Rate:    {dep_opt:.1f} nm/min")
print(f"  Stress:       {stress_opt:.1f} MPa")
print(f"  Ref. Index:   {n_opt:.4f}")
print(f"  Uniformity:   {unif_opt:.2f} %")
print(f"  Desirability: {-best_score:.3f} (1.0 = perfect)")

# ── 6. PARETO FRONT PLOT ──────────────────────────────────────────────────────
# Shows tradeoff between dep rate and stress
# Engineers must choose: fast deposition OR low stress, rarely both

print("Generating Pareto front analysis...")

dep_rates, stresses, n_indices, uniformities = [], [], [], []
for _ in range(300):
    T_r  = np.random.uniform(250, 350)
    P_r  = np.random.uniform(50, 150)
    Pg_r = np.random.uniform(1, 3)
    Q_r  = np.random.uniform(20, 60)
    d, s, n, u = pecvd_model(T_r, P_r, Pg_r, Q_r)
    dep_rates.append(d); stresses.append(s)
    n_indices.append(n); uniformities.append(u)

dep_rates  = np.array(dep_rates)
stresses   = np.array(stresses)
n_indices  = np.array(n_indices)
uniformities = np.array(uniformities)

# Color by refractive index quality
n_quality = np.abs(n_indices - 1.95)
in_spec = n_quality < 0.1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Dep Rate vs Stress tradeoff
sc = axes[0].scatter(dep_rates[~in_spec], np.abs(stresses[~in_spec]),
                     c='lightgray', alpha=0.4, s=20, label='Out of spec (n)')
sc2 = axes[0].scatter(dep_rates[in_spec], np.abs(stresses[in_spec]),
                      c=uniformities[in_spec], cmap='RdYlGn_r',
                      alpha=0.8, s=40, label='In spec (n: 1.85-2.05)')
axes[0].scatter(dep_opt, abs(stress_opt), c='red', s=200, 
                marker='*', zorder=5, label=f'Optimum ({dep_opt:.0f} nm/min, {abs(stress_opt):.0f} MPa)')
axes[0].set_xlabel('Deposition Rate (nm/min)')
axes[0].set_ylabel('|Stress| (MPa)')
axes[0].set_title('Process Tradeoff: Dep Rate vs Stress\n'
                  'Color = Uniformity (green=good)', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes[0], label='Uniformity (%)')

# Plot 2: Process window contour
T_2d  = np.linspace(250, 350, 40)
P_2d  = np.linspace(50, 150, 40)
TT, PP = np.meshgrid(T_2d, P_2d)
desir = np.zeros_like(TT)

for i in range(TT.shape[0]):
    for j in range(TT.shape[1]):
        params = [TT[i,j], PP[i,j], 2.0, 40.0]
        desir[i,j] = -objective_function(params)

cp = axes[1].contourf(TT, PP, desir, levels=15, cmap='RdYlGn')
axes[1].contour(TT, PP, desir, levels=[0.5, 0.65, 0.75], 
                colors='black', linewidths=0.8, linestyles='--')
axes[1].scatter(T_opt, P_opt, c='red', s=200, marker='*', 
                zorder=5, label=f'Optimum\n({T_opt:.0f}°C, {P_opt:.0f}W)')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('RF Power (W)')
axes[1].set_title('Process Window Map\n'
                  'Green = High Desirability', fontweight='bold')
axes[1].legend(fontsize=9)
plt.colorbar(cp, ax=axes[1], label='Desirability Score')

plt.suptitle('PECVD Si₃N₄ Multi-Response Optimization\n'
             'Pressure=2 Torr, SiH4=40 sccm',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module4_optimization.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/module4_optimization.png")

print("\nModule 4 complete.")