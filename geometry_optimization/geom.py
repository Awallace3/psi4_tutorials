import psi4
import numpy as np
from pprint import pprint as pp
import pandas as pd
import matplotlib.pyplot as plt
import os

# Setting psi4 resources (Memory and threads)
psi4.set_memory("124 GB")
psi4.set_num_threads(24)

# Set the output file
psi4.set_output_file("geom.out", False)

# Redefine the molecule (optionally with worse guess geometry)
h2o = psi4.geometry("""
0 1
O   0.0   0.0   0.0
H   0.0   1.5   0.0
H   1.5   0.9   0.0
""")
# Set new options
psi4.set_options({
    'basis': 'def2-svp',
    'scf_type': 'df',
    'geom_maxiter': 500
})

# Run geometry optimization
# opt_energy = psi4.optimize('b3lyp', molecule=h2o, engine='geometric')
opt_energy = psi4.optimize('b3lyp', molecule=h2o)
print(f"Optimized B3LYP/def2-SVP Energy: {opt_energy:.6f} Hartree")

# Print final optimized geometry
print("Optimized Geometry (Angstrom):")
print(h2o.save_string_xyz())
with open('final_opt.xyz', 'w') as f:
    f.write(str(h2o.natom()) + "\n")
    f.write(h2o.save_string_xyz())

# To ensure we have a minimum, we can check the Hessian through vibrational
# frequency analysis
energy, wfn_b3lyp = psi4.energy('b3lyp', molecule=h2o, return_wfn=True)
freq_e, wfn_b3lyp = psi4.frequency("b3lyp", molecule=h2o, return_wfn=True)
vibinfo_b3lyp = wfn_b3lyp.frequency_analysis
freq_e_hf, wfn_hf = psi4.frequency("hf/sto-3g", molecule=h2o, return_wfn=True)
vibinfo_hf = wfn_hf.frequency_analysis

# 3N-6(Nonlinear) or 3N-5(Linear)
data = {
    "B3LYP/def2-svp Frequency (cm^-1)": vibinfo_b3lyp['omega'].data,
    "B3LYP/def2-svp IR Intensity": vibinfo_b3lyp['IR_intensity'].data,
    "HF/sto-3g Frequency (cm^-1)": vibinfo_hf['omega'].data,
    "HF/sto-3g IR Intensity": vibinfo_hf['IR_intensity'].data,
}
df = pd.DataFrame(data)
df['B3LYP/def2-svp Frequency (cm^-1) real'] = df.apply(lambda r: r['B3LYP/def2-svp Frequency (cm^-1)'].real, axis=1)
df['B3LYP/def2-svp Frequency (cm^-1) imag'] = df.apply(lambda r: r['B3LYP/def2-svp Frequency (cm^-1)'].imag, axis=1)
print(df[['B3LYP/def2-svp Frequency (cm^-1) real', 'B3LYP/def2-svp Frequency (cm^-1) imag']])

df['HF/sto-3g Frequency (cm^-1) real'] = df.apply(lambda r: r['HF/sto-3g Frequency (cm^-1)'].real, axis=1)
df['HF/sto-3g Frequency (cm^-1) imag'] = df.apply(lambda r: r['HF/sto-3g Frequency (cm^-1)'].imag, axis=1)
print(df[['HF/sto-3g Frequency (cm^-1) real', 'HF/sto-3g Frequency (cm^-1) imag']])

# Drop all rows that have imaginary frequencies not equal to 0
df.sort_values(by='B3LYP/def2-svp Frequency (cm^-1) real', inplace=True, ascending=False)
n_atoms = h2o.natom()
df = df.head(n_atoms)

# Assert that all imaginary frequencies are 0, ensuring that we have a minimum
assert df['B3LYP/def2-svp Frequency (cm^-1) imag'].sum() == 0


df['Experimental Frequency (cm^-1)'] = [3585, 3506, 1885]
df['Experimental IR Intensity'] = [0.17, 1.0, 0.15]

print(df[['B3LYP/def2-svp Frequency (cm^-1) real', 'HF/sto-3g Frequency (cm^-1) real', 'Experimental Frequency (cm^-1)']])

# Normalize Computational IR Intensity
df['B3LYP/def2-svp IR Intensity'] = df['B3LYP/def2-svp IR Intensity'] / df['B3LYP/def2-svp IR Intensity'].max()
df['HF/sto-3g IR Intensity'] = df['HF/sto-3g IR Intensity'] / df['HF/sto-3g IR Intensity'].max()

df['B3LYP/def2-svp Error (cm^-1)'] = df['B3LYP/def2-svp Frequency (cm^-1) real'] - df['Experimental Frequency (cm^-1)']
df['HF/sto-3g Error (cm^-1)'] = df['HF/sto-3g Frequency (cm^-1) real'] - df['Experimental Frequency (cm^-1)']

# Simple Error Statistics
print(df[['B3LYP/def2-svp Error (cm^-1)', 'HF/sto-3g Error (cm^-1)']].describe())

# Use DF to create artificial spectra that has sticks at the frequencies with x-axis of 0-5000 cm^-1
# Create the stick spectrum
fig, ax = plt.subplots(figsize=(8, 4))
ax.vlines(df["B3LYP/def2-svp Frequency (cm^-1) real"], 0, df["B3LYP/def2-svp IR Intensity"], colors='blue', linewidth=1.2, label='B3LYP/def2-svp')
ax.vlines(df["HF/sto-3g Frequency (cm^-1) real"], 0, df["HF/sto-3g IR Intensity"], colors='red', linewidth=1.2, label='HF/sto-3g')
ax.vlines(df["Experimental Frequency (cm^-1)"], 0, df["Experimental IR Intensity"], colors='black', linewidth=1.2, label='Experimental')

# Aesthetic details
ax.set_xlim(0, 5000)
ax.set_ylim(0, df["B3LYP/def2-svp IR Intensity"].max() * 1.1)
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("IR Intensity (arbitrary units)")
ax.set_title("Simulated IR Spectrum (Stick Plot)")
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("stick_spectrum.png", dpi=300)


# visualize the geometry optimization
import py3Dmol
import urllib
os.system("./extract_geoms_from_opt.awk geom.log > opt_trajectory.xyz")


with open("opt_trajectory.xyz") as f:
    xyz = f.read()

view = py3Dmol.view(width=500, height=400)
view.addModelsAsFrames(xyz, "xyz")
view.setStyle({'stick': {}})
view.animate({'loop': 'backAndForth'})
view.zoomTo()
html = view._make_html()
with open("opt_trajectory.html", "w") as f:
    f.write(html)
os.system("open opt_trajectory.html")
