import psi4
import numpy as np
from pprint import pprint as pp
import pandas as pd
import matplotlib.pyplot as plt
import os

# Setting psi4 resources (Memory and threads)
psi4.set_memory("64 GB")
psi4.set_num_threads(24)

# Set the output file
psi4.set_output_file("geom.out", False)

def h2o():
    # Redefine the molecule (optionally with worse guess geometry)
    mol = psi4.geometry("""
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
    opt_energy = psi4.optimize('b3lyp', molecule=mol)
    print(f"Optimized B3LYP/def2-SVP Energy: {opt_energy:.6f} Hartree")

    # Print final optimized geometry
    print("Optimized Geometry (Angstrom):")
    print(mol.save_string_xyz())
    with open('final_opt.xyz', 'w') as f:
        f.write(str(mol.natom()) + "\n")
        f.write(mol.save_string_xyz())

    # To ensure we have a minimum, we can check the Hessian through vibrational
    # frequency analysis
    energy, wfn_b3lyp = psi4.energy('b3lyp', molecule=mol, return_wfn=True)
    freq_e, wfn_b3lyp = psi4.frequency("b3lyp", molecule=mol, return_wfn=True)
    vibinfo_b3lyp = wfn_b3lyp.frequency_analysis
    freq_e_hf, wfn_hf = psi4.frequency("hf/sto-3g", molecule=mol, return_wfn=True)
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
    n_atoms = mol.natom()
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
    import re

    # Extract geometries from optimization log
    os.system("./extract_geoms_from_opt.awk geom.log > opt_trajectory.xyz")

    # Function to interpolate between two geometries
    def interpolate_geometries(geom1, geom2, step1, step2, num_frames=10):
        """
        Interpolate between two geometries to create a smooth transition.
        
        Parameters:
        geom1, geom2: Lists of [atom_symbol, x, y, z] for each geometry
        step1, step2: Step numbers for the geometries
        num_frames: Number of interpolated frames to generate
        
        Returns:
        List of interpolated geometries with their step labels
        """
        interpolated = []
        
        # Ensure both geometries have the same number of atoms
        assert len(geom1) == len(geom2), "Geometries must have the same number of atoms"
        
        for i in range(1, num_frames):  # Skip first and last as they are the original geometries
            # Calculate interpolation factor (0 to 1)
            t = i / num_frames
            
            # Create interpolated geometry
            new_geom = []
            for atom_idx in range(len(geom1)):
                atom_symbol = geom1[atom_idx][0]  # Symbol should be the same in both geometries
                
                # Linear interpolation of coordinates
                x = geom1[atom_idx][1] + t * (geom2[atom_idx][1] - geom1[atom_idx][1])
                y = geom1[atom_idx][2] + t * (geom2[atom_idx][2] - geom1[atom_idx][2])
                z = geom1[atom_idx][3] + t * (geom2[atom_idx][3] - geom1[atom_idx][3])
                
                new_geom.append([atom_symbol, x, y, z])
            
            # Create step label for interpolated frame
            step_label = f"Step {step1}-{step2} Interpolated {i}/{num_frames-1}"
            interpolated.append((new_geom, step_label))
        
        return interpolated

    # Read the XYZ file directly
    with open("opt_trajectory.xyz") as f:
        lines = f.readlines()

    # Parse the XYZ file to extract geometries
    geometries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():  # Number of atoms line
            num_atoms = int(lines[i].strip())
            step_line = lines[i+1].strip()  # Step X line
            
            # Extract step number
            step_num = int(step_line.split()[1]) if len(step_line.split()) > 1 else 0
            
            # Read atom coordinates
            atoms = []
            for j in range(i+2, i+2+num_atoms):
                if j < len(lines):
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms.append([symbol, x, y, z])
            
            geometries.append((atoms, step_num))
            i += num_atoms + 2  # Move to the next geometry
        else:
            i += 1  # Skip any unexpected lines

    # Create interpolated trajectory
    interpolated_xyz = ""
    for i in range(len(geometries) - 1):
        current_geom, current_step = geometries[i]
        next_geom, next_step = geometries[i+1]
        
        # Add the current geometry
        interpolated_xyz += f"{len(current_geom)}\n"
        interpolated_xyz += f"Step {current_step}\n"
        for atom in current_geom:
            symbol, x, y, z = atom
            interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
        
        # Add interpolated geometries
        interpolated_frames = interpolate_geometries(current_geom, next_geom, current_step, next_step, num_frames=10)
        
        for frame, step_label in interpolated_frames:
            interpolated_xyz += f"{len(frame)}\n"
            interpolated_xyz += f"{step_label}\n"
            for atom in frame:
                symbol, x, y, z = atom
                interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"

    # Add the last geometry
    last_geom, last_step = geometries[-1]
    interpolated_xyz += f"{len(last_geom)}\n"
    interpolated_xyz += f"Step {last_step}\n"
    for atom in last_geom:
        symbol, x, y, z = atom
        interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"

    # Save the interpolated trajectory
    with open("interpolated_trajectory.xyz", "w") as f:
        f.write(interpolated_xyz.strip())

    # Count the number of frames in the interpolated trajectory
    frame_count = interpolated_xyz.count('\n3\n')  # Count occurrences of atom count lines

    # Visualize with py3Dmol
    view = py3Dmol.view(width=500, height=400)
    view.addModelsAsFrames(interpolated_xyz, "xyz")
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
    view.animate({'loop': 'forward'})
    view.zoomTo()
    view.show()

    # Optionally save as HTML for external viewing
    html = view._make_html()
    with open("opt_trajectory.html", "w") as f:
        f.write(html)

    print(f"Created smooth trajectory with {frame_count} frames")
    # Uncomment to open in browser automatically
    os.system("firefox opt_trajectory.html")


def h2co3():
    # Redefine the molecule (optionally with worse guess geometry)
    mol = psi4.geometry("""
    0 1
    O   0.0   0.0   0.0
    H   0.0   1.5   0.0
    H   1.5   0.9   0.0
    """)
    mol = psi4.geometry("""
    0 1
    C 0.00000000 0.00000000 0.00000000
    O -1.27746485 -0.62401263 0.22452493
    H -1.67988085 -0.62978463 -0.62652407
    O -0.01959185 1.64774837 0.14399293
    O 1.28129615 -0.57982263 0.45071493
    H 1.05666915 -1.38296363 0.88818893
    """)
    # Set new options
    psi4.set_options({
        'basis': 'def2-svp',
        'scf_type': 'df',
        'geom_maxiter': 500
    })

    # Run geometry optimization
    # opt_energy = psi4.optimize('b3lyp', molecule=h2o, engine='geometric')
    opt_energy = psi4.optimize('b3lyp', molecule=mol)
    print(f"Optimized B3LYP/def2-SVP Energy: {opt_energy:.6f} Hartree")

    # Print final optimized geometry
    print("Optimized Geometry (Angstrom):")
    print(mol.save_string_xyz())
    with open('final_opt_h2co3.xyz', 'w') as f:
        f.write(str(mol.natom()) + "\n")
        f.write(mol.save_string_xyz())

    # To ensure we have a minimum, we can check the Hessian through vibrational
    # frequency analysis
    energy, wfn_b3lyp = psi4.energy('b3lyp', molecule=mol, return_wfn=True)
    freq_e, wfn_b3lyp = psi4.frequency("b3lyp", molecule=mol, return_wfn=True)
    vibinfo_b3lyp = wfn_b3lyp.frequency_analysis
    freq_e_hf, wfn_hf = psi4.frequency("hf/sto-3g", molecule=mol, return_wfn=True)
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
    n_atoms = mol.natom()
    df = df.head(n_atoms)

    # Assert that all imaginary frequencies are 0, ensuring that we have a minimum
    assert df['B3LYP/def2-svp Frequency (cm^-1) imag'].sum() == 0

    # Normalize Computational IR Intensity
    df['B3LYP/def2-svp IR Intensity'] = df['B3LYP/def2-svp IR Intensity'] / df['B3LYP/def2-svp IR Intensity'].max()
    df['HF/sto-3g IR Intensity'] = df['HF/sto-3g IR Intensity'] / df['HF/sto-3g IR Intensity'].max()

    # Use DF to create artificial spectra that has sticks at the frequencies with x-axis of 0-5000 cm^-1
    # Create the stick spectrum
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.vlines(df["B3LYP/def2-svp Frequency (cm^-1) real"], 0, df["B3LYP/def2-svp IR Intensity"], colors='blue', linewidth=1.2, label='B3LYP/def2-svp')
    ax.vlines(df["HF/sto-3g Frequency (cm^-1) real"], 0, df["HF/sto-3g IR Intensity"], colors='red', linewidth=1.2, label='HF/sto-3g')

    # Aesthetic details
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, df["B3LYP/def2-svp IR Intensity"].max() * 1.1)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("IR Intensity (arbitrary units)")
    ax.set_title("Simulated IR Spectrum (Stick Plot)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("stick_spectrum_h2co3.png", dpi=300)


    # visualize the geometry optimization
    import py3Dmol
    import re

    # Extract geometries from optimization log
    os.system("./extract_geoms_from_opt.awk geom.log > opt_trajectory_h2co3.xyz")

    # Function to interpolate between two geometries
    def interpolate_geometries(geom1, geom2, step1, step2, num_frames=10):
        """
        Interpolate between two geometries to create a smooth transition.
        
        Parameters:
        geom1, geom2: Lists of [atom_symbol, x, y, z] for each geometry
        step1, step2: Step numbers for the geometries
        num_frames: Number of interpolated frames to generate
        
        Returns:
        List of interpolated geometries with their step labels
        """
        interpolated = []
        
        # Ensure both geometries have the same number of atoms
        assert len(geom1) == len(geom2), "Geometries must have the same number of atoms"
        
        for i in range(1, num_frames):  # Skip first and last as they are the original geometries
            # Calculate interpolation factor (0 to 1)
            t = i / num_frames
            
            # Create interpolated geometry
            new_geom = []
            for atom_idx in range(len(geom1)):
                atom_symbol = geom1[atom_idx][0]  # Symbol should be the same in both geometries
                
                # Linear interpolation of coordinates
                x = geom1[atom_idx][1] + t * (geom2[atom_idx][1] - geom1[atom_idx][1])
                y = geom1[atom_idx][2] + t * (geom2[atom_idx][2] - geom1[atom_idx][2])
                z = geom1[atom_idx][3] + t * (geom2[atom_idx][3] - geom1[atom_idx][3])
                
                new_geom.append([atom_symbol, x, y, z])
            
            # Create step label for interpolated frame
            step_label = f"Step {step1}-{step2} Interpolated {i}/{num_frames-1}"
            interpolated.append((new_geom, step_label))
        
        return interpolated

    # Read the XYZ file directly
    with open("opt_trajectory_h2co3.xyz") as f:
        lines = f.readlines()

    # Parse the XYZ file to extract geometries
    geometries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():  # Number of atoms line
            num_atoms = int(lines[i].strip())
            step_line = lines[i+1].strip()  # Step X line
            
            # Extract step number
            step_num = int(step_line.split()[1]) if len(step_line.split()) > 1 else 0
            
            # Read atom coordinates
            atoms = []
            for j in range(i+2, i+2+num_atoms):
                if j < len(lines):
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        atoms.append([symbol, x, y, z])
            
            geometries.append((atoms, step_num))
            i += num_atoms + 2  # Move to the next geometry
        else:
            i += 1  # Skip any unexpected lines

    # Create interpolated trajectory
    interpolated_xyz = ""
    for i in range(len(geometries) - 1):
        current_geom, current_step = geometries[i]
        next_geom, next_step = geometries[i+1]
        
        # Add the current geometry
        interpolated_xyz += f"{len(current_geom)}\n"
        interpolated_xyz += f"Step {current_step}\n"
        for atom in current_geom:
            symbol, x, y, z = atom
            interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
        
        # Add interpolated geometries
        interpolated_frames = interpolate_geometries(current_geom, next_geom, current_step, next_step, num_frames=10)
        
        for frame, step_label in interpolated_frames:
            interpolated_xyz += f"{len(frame)}\n"
            interpolated_xyz += f"{step_label}\n"
            for atom in frame:
                symbol, x, y, z = atom
                interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"

    # Add the last geometry
    last_geom, last_step = geometries[-1]
    interpolated_xyz += f"{len(last_geom)}\n"
    interpolated_xyz += f"Step {last_step}\n"
    for atom in last_geom:
        symbol, x, y, z = atom
        interpolated_xyz += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"

    # Save the interpolated trajectory
    with open("interpolated_trajectory_h2co3.xyz", "w") as f:
        f.write(interpolated_xyz.strip())

    # Count the number of frames in the interpolated trajectory
    frame_count = interpolated_xyz.count('\n3\n')  # Count occurrences of atom count lines

    # Visualize with py3Dmol
    view = py3Dmol.view(width=500, height=400)
    view.addModelsAsFrames(interpolated_xyz, "xyz")
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
    view.animate({'loop': 'forward'})
    view.zoomTo()
    view.show()

    # Optionally save as HTML for external viewing
    html = view._make_html()
    with open("opt_trajectory_h2co3.html", "w") as f:
        f.write(html)

    print(f"Created smooth trajectory with {frame_count} frames")
    # Uncomment to open in browser automatically
    os.system("firefox opt_trajectory_h2co3.html")
    # Note that this optimized geometry falls into a minimum (Syn-Anti), but
    # the global minimum is actually a Syn-Syn conformation.
    # https://pubs.acs.org/doi/full/10.1021/acs.jpca.1c02878


def main():
    h2o()
    h2co3()
    return


if __name__ == "__main__":
    main()
