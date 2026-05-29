import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

# ── WHAT THIS MODULE DOES ─────────────────────────────────────────────────────
# Simulates spectroscopic ellipsometry for semiconductor thin films
# Ellipsometry measures how polarized light changes when it reflects off a 
# thin film stack. From that change we extract: film thickness, n (refractive 
# index), k (extinction coefficient). This is how fabs verify CVD/PVD/ALD 
# deposition thickness non-destructively in-line.
# Tools like this run on KLA, AMAT, and ASM metrology equipment every day.

os.makedirs('outputs', exist_ok=True)

# ── 1. MATERIAL OPTICAL CONSTANTS ─────────────────────────────────────────────
# n = refractive index (how much light slows down in material)
# k = extinction coefficient (how much light is absorbed)
# These are real published values for common semiconductor materials

WAVELENGTHS = np.linspace(300, 800, 200)  # 300-800nm visible + UV range

def get_optical_constants(material, wavelengths):
    """
    Returns n, k vs wavelength for common semiconductor materials.
    Values based on published literature (Palik Handbook of Optical Constants).
    """
    w = wavelengths
    
    if material == 'SiO2':
        # Silicon dioxide: transparent in visible, low n
        n = 1.45 + 0.003 * (500/w)**2
        k = np.zeros_like(w)
        
    elif material == 'Si3N4':
        # Silicon nitride: used as passivation, etch stop layers
        n = 2.0 + 0.01 * (500/w)**2
        k = np.where(w < 350, 0.1 * (350-w)/50, 0)
        
    elif material == 'Si':
        # Silicon substrate: absorbs strongly in UV
        n = 3.5 + 2.0 * np.exp(-(w-350)**2 / (2*80**2))
        k = np.where(w < 400, 0.1 + 2*(400-w)/400, 
            np.where(w < 500, 0.1*(500-w)/100, 0.001))
        
    elif material == 'TiN':
        # Titanium nitride: used as diffusion barrier, electrode
        n = 1.5 + 2.0*(w/500)**0.5
        k = 1.5 + 1.0*(w/500)**0.3
        
    elif material == 'Al2O3':
        # Alumina: ALD high-k dielectric
        n = 1.63 + 0.005*(500/w)**2
        k = np.zeros_like(w)
        
    elif material == 'underfill':
        # Epoxy underfill for advanced packaging (Henkel-type)
        n = 1.52 + 0.004*(500/w)**2
        k = np.where(w < 380, 0.05*(380-w)/80, 0)
        
    else:
        n = np.ones_like(w) * 1.5
        k = np.zeros_like(w)
    
    return n, k

# ── 2. TRANSFER MATRIX METHOD ─────────────────────────────────────────────────
# The TMM calculates how light propagates through a stack of thin films
# Each layer is represented as a 2x2 matrix
# Multiply all matrices together to get total reflection/transmission

def tmm_reflectance(wavelengths, layers, theta_i=0.0):
    """
    Calculate reflectance spectrum for a multilayer thin film stack.
    
    layers: list of (material, thickness_nm) tuples
            Last layer is always the substrate (thickness ignored)
    theta_i: angle of incidence in degrees
    
    Returns: reflectance array (0-1) for each wavelength
    """
    theta_i_rad = np.deg2rad(theta_i)
    R_spectrum = np.zeros(len(wavelengths))
    
    for idx, wl in enumerate(wavelengths):
        # Build N array: complex refractive index for each layer
        N = []
        for material, thickness in layers:
            n, k = get_optical_constants(material, np.array([wl]))
            N.append(complex(n[0], -k[0]))
        
        # Transfer matrix calculation (s-polarization)
        # Phase thickness for each layer
        M_total = np.eye(2, dtype=complex)
        
        n0 = N[0]  # incident medium (usually air, n=1)
        cos_t0 = np.sqrt(1 - (n0.real * np.sin(theta_i_rad) / N[0])**2)
        
        for i in range(1, len(layers)-1):
            ni = N[i]
            thickness_nm = layers[i][1]
            
            # Snell's law for angle in this layer
            sin_ti = (N[0].real * np.sin(theta_i_rad)) / ni
            cos_ti = np.sqrt(1 - sin_ti**2 + 0j)
            
            # Phase thickness (how much the wave accumulates crossing the layer)
            delta = 2 * np.pi * ni * cos_ti * thickness_nm / wl
            
            # Layer transfer matrix
            Mi = np.array([
                [np.cos(delta), -1j * np.sin(delta) / (ni * cos_ti)],
                [-1j * ni * cos_ti * np.sin(delta), np.cos(delta)]
            ], dtype=complex)
            
            M_total = M_total @ Mi
        
        # Substrate
        ns = N[-1]
        sin_ts = (N[0].real * np.sin(theta_i_rad)) / ns
        cos_ts = np.sqrt(1 - sin_ts**2 + 0j)
        
        # Fresnel coefficients from total matrix
        m11, m12, m21, m22 = M_total[0,0], M_total[0,1], M_total[1,0], M_total[1,1]
        
        eta_s = ns * cos_ts
        eta_0 = N[0] * cos_t0
        
        denom = m11*eta_0 + m12*eta_s*eta_0 + m21 + m22*eta_s
        numer = m11*eta_0 + m12*eta_s*eta_0 - m21 - m22*eta_s
        
        if abs(denom) > 1e-10:
            r = numer / denom
            R_spectrum[idx] = abs(r)**2
        else:
            R_spectrum[idx] = 0
    
    return R_spectrum

# ── 3. SIMULATE FOUR REAL SEMICONDUCTOR FILM STACKS ───────────────────────────
print("Simulating thin film reflectance spectra...")

stacks = {
    'SiO2 on Si\n(Gate oxide, 100nm)': [
        ('air', 0), ('SiO2', 100), ('Si', 0)
    ],
    'Si3N4 on Si\n(Passivation, 200nm)': [
        ('air', 0), ('Si3N4', 200), ('Si', 0)
    ],
    'TiN on Si\n(Barrier layer, 50nm)': [
        ('air', 0), ('TiN', 50), ('Si', 0)
    ],
    'Al2O3 on Si\n(ALD High-k, 10nm)': [
        ('air', 0), ('Al2O3', 10), ('Si', 0)
    ],
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']

for idx, (name, stack) in enumerate(stacks.items()):
    R = tmm_reflectance(WAVELENGTHS, stack)
    axes[idx].plot(WAVELENGTHS, R, color=colors[idx], linewidth=2)
    axes[idx].set_title(name, fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Wavelength (nm)')
    axes[idx].set_ylabel('Reflectance')
    axes[idx].set_ylim(0, 1)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].fill_between(WAVELENGTHS, R, alpha=0.15, color=colors[idx])

plt.suptitle('Optical Reflectance Spectra — Semiconductor Thin Film Stacks\n'
             'Simulated via Transfer Matrix Method (TMM)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module2_reflectance_spectra.png', dpi=150)
plt.close()
print("Saved: outputs/module2_reflectance_spectra.png")

# ── 4. THICKNESS UNIFORMITY MAP ───────────────────────────────────────────────
# In a real fab, metrology tools measure film thickness at ~50-100 points
# across a 300mm wafer. Non-uniformity = process problem.
# We simulate a realistic PVD deposition with edge effects.

print("Generating wafer thickness uniformity map...")

# 300mm wafer grid
wafer_radius = 150  # mm
x = np.linspace(-150, 150, 80)
y = np.linspace(-150, 150, 80)
XX, YY = np.meshgrid(x, y)
R_wafer = np.sqrt(XX**2 + YY**2)

# Simulate SiO2 thickness: target 100nm, edge thinning + center hotspot
# This is realistic for a CVD process with slight center-to-edge gradient
np.random.seed(42)
thickness_map = (
    100                                          # target thickness
    - 8 * (R_wafer / wafer_radius)**2            # edge thinning (parabolic)
    + 3 * np.exp(-R_wafer**2 / (2*40**2))       # center hotspot
    + np.random.normal(0, 0.8, XX.shape)         # measurement noise
)

# Mask outside wafer
mask = R_wafer > wafer_radius
thickness_map[mask] = np.nan

# Calculate uniformity metric (industry standard: 1-sigma / mean * 100)
valid = thickness_map[~np.isnan(thickness_map)]
uniformity = (valid.std() / valid.mean()) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Thickness map
im = axes[0].contourf(XX, YY, thickness_map, levels=20, cmap='RdYlGn_r')
circle = plt.Circle((0,0), wafer_radius, fill=False, color='black', linewidth=2)
axes[0].add_patch(circle)
axes[0].set_aspect('equal')
axes[0].set_title(f'SiO2 Thickness Map — 300mm Wafer\n'
                  f'Target: 100nm | 1σ Uniformity: {uniformity:.2f}%', 
                  fontweight='bold')
axes[0].set_xlabel('X position (mm)')
axes[0].set_ylabel('Y position (mm)')
plt.colorbar(im, ax=axes[0], label='Thickness (nm)')

# Histogram of thickness distribution
axes[1].hist(valid, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[1].axvline(valid.mean(), color='red', linewidth=2, 
                label=f'Mean: {valid.mean():.1f} nm')
axes[1].axvline(valid.mean() + valid.std(), color='orange', 
                linewidth=1.5, linestyle='--', label=f'+1σ: {valid.mean()+valid.std():.1f} nm')
axes[1].axvline(valid.mean() - valid.std(), color='orange', 
                linewidth=1.5, linestyle='--', label=f'-1σ: {valid.mean()-valid.std():.1f} nm')
axes[1].set_xlabel('Thickness (nm)')
axes[1].set_ylabel('Count')
axes[1].set_title('Thickness Distribution\nAcross Wafer')
axes[1].legend()

plt.suptitle('Thin Film Thickness Uniformity Analysis', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module2_uniformity_map.png', dpi=150)
plt.close()
print("Saved: outputs/module2_uniformity_map.png")

# ── 5. ELLIPSOMETRY THICKNESS FITTING ─────────────────────────────────────────
# Real ellipsometry: measure Psi and Delta (polarization angles)
# Then fit a model to extract thickness
# We simulate a measurement then recover the thickness via optimization

print("Running ellipsometry thickness fitting...")

def simulate_ellipsometry(wavelengths, thickness_nm, material='SiO2'):
    """Simulate Psi (amplitude ratio) and Delta (phase difference)"""
    layers = [('air', 0), (material, thickness_nm), ('Si', 0)]
    
    # Calculate at two angles (standard SE measurement)
    R_s_65 = tmm_reflectance(wavelengths, layers, theta_i=65)
    R_p_65 = tmm_reflectance(wavelengths, layers, theta_i=75)
    
    # Psi and Delta (simplified from full Jones matrix)
    ratio = np.sqrt(R_p_65 / (R_s_65 + 1e-10))
    Psi = np.degrees(np.arctan(ratio))
    Delta = np.angle(np.exp(1j * 2 * np.pi * 
                    np.array([t/500 for t in wavelengths]))) * 180 / np.pi + 180
    
    return Psi, Delta

# Simulate "measured" data at true thickness 142nm
true_thickness = 142.0
Psi_meas, Delta_meas = simulate_ellipsometry(WAVELENGTHS, true_thickness)
# Add realistic measurement noise
Psi_meas += np.random.normal(0, 0.3, len(WAVELENGTHS))
Delta_meas += np.random.normal(0, 0.5, len(WAVELENGTHS))

# Fit: find thickness that best matches measured spectra
def objective(t):
    Psi_fit, Delta_fit = simulate_ellipsometry(WAVELENGTHS, t[0])
    return np.sum((Psi_fit - Psi_meas)**2 + 0.1*(Delta_fit - Delta_meas)**2)

result = minimize(objective, x0=[120.0], method='Nelder-Mead',
                  options={'xatol': 0.1, 'fatol': 1e-4, 'maxiter': 500})
fitted_thickness = result.x[0]

print(f"  True thickness:   {true_thickness:.1f} nm")
print(f"  Fitted thickness: {fitted_thickness:.1f} nm")
print(f"  Error:            {abs(fitted_thickness - true_thickness):.2f} nm")

# Plot ellipsometry fit
Psi_fitted, Delta_fitted = simulate_ellipsometry(WAVELENGTHS, fitted_thickness)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(WAVELENGTHS, Psi_meas, 'o', markersize=2, 
             color='steelblue', alpha=0.6, label='Measured')
axes[0].plot(WAVELENGTHS, Psi_fitted, 'r-', linewidth=2, 
             label=f'Model fit ({fitted_thickness:.1f} nm)')
axes[0].set_xlabel('Wavelength (nm)')
axes[0].set_ylabel('Ψ (degrees)')
axes[0].set_title('Ellipsometry — Psi Spectrum')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(WAVELENGTHS, Delta_meas, 'o', markersize=2, 
             color='steelblue', alpha=0.6, label='Measured')
axes[1].plot(WAVELENGTHS, Delta_fitted, 'r-', linewidth=2, 
             label=f'Model fit ({fitted_thickness:.1f} nm)')
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('Δ (degrees)')
axes[1].set_title('Ellipsometry — Delta Spectrum')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Spectroscopic Ellipsometry Fit — SiO2 on Si\n'
             f'True: {true_thickness}nm | Fitted: {fitted_thickness:.1f}nm | '
             f'Error: {abs(fitted_thickness-true_thickness):.2f}nm',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module2_ellipsometry_fit.png', dpi=150)
plt.close()
print("Saved: outputs/module2_ellipsometry_fit.png")

print("\nModule 2 complete.")