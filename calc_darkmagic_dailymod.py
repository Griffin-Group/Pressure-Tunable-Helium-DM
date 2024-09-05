import sys

import numpy as np
from darkmagic.benchmark_models import (
    heavy_scalar_mediator,
    light_scalar_mediator,
)
from darkmagic.calculator import Calculator
from darkmagic.material import MaterialParameters, PhononMaterial
from darkmagic.numerics import Numerics
from mpi4py.MPI import COMM_WORLD as comm


def run_calc(masses, times, material, model):
    numerics = Numerics(
        N_grid=[80, 40, 40],
        N_DWF_grid=[30, 30, 30],
        use_special_mesh=bool(model.shortname == "lsm"),
        use_q_cut=True,
    )
    model = light_scalar_mediator
    if comm.Get_rank() == 0:
        print("Model: {model.name}")
    full_calc = Calculator("scattering", masses, material, model, numerics, times)
    full_calc.evaluate(mpi=True)

    # Save results
    hdf5_filename = f"DarkMAGIC_{material.name}_{model.shortname}_dailymod.hdf5"
    full_calc.to_file(hdf5_filename)


if __name__ == "__main__":
    # Masses and times
    masses = np.logspace(3, 10, 96)
    times = np.arange(0, 25, 1)

    pressure = int(sys.argv[1])  # Pressure in GPa
    phase = sys.argv[2]  # fcc or hcp

    n_atoms = 2 if phase == "fcc" else 1
    yaml_file = f"p{pressure*10:03d}/phonopy.yaml"
    material_name = f"{phase}_He_{pressure:02d}GPa"
    params = MaterialParameters(
        N={"e": [2] * n_atoms, "n": [2] * n_atoms, "p": [2] * n_atoms}
    )
    material = PhononMaterial(material_name, params, yaml_file)

    run_calc(masses, times, material, light_scalar_mediator)
