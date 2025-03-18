import glob
import os
import subprocess
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import yaml
from ase.formula import Formula
from ase.io import read, write

app = typer.Typer(help="CLI for setting up and running MC simulations with SLURM.")


EQ_CONF = {
    "ensemble": "NPT",
    "steps": 100000,
    "temperature": 300.0,
    "pressure": 1.0,
    "mace_model": "UNK",
    "system": "UNK",
    "output files": {
        "thermo": "EQ/thermo.csv",
        "trajectory": "EQ/trj.xyz",
        "events": "EQ/events.csv",
        "restart": "EQ/restart.xyz",
    },
    "saving step": {
        "thermo": 10,
        "trajectory": 50,
        "events": 100,
        "states": 1000,
        "restart": 1000,
    },
    "states folder": "EQ/states",
    "probabilities": {"position": 0.5, "volume": 0.5},
    "tuning every": 500,
    "continue": False,
    "cutoff": 6.0,
    "chemical potential": {},
    "seed": None,
    "max displacement": 0.175,
    "max volume change": 0.01,
}

PID_MC_CONF = {
    "ensemble": "muPT",
    "steps": 1000000,
    "temperature": 300.0,
    "pressure": 1.0,
    "mace_model": "UNK",
    "system": "UNK",
    "output files": {
        "thermo": "PID/thermo.csv",
        "trajectory": "PID/trj.xyz",
        "events": "PID/events.csv",
        "restart": "PID/restart.xyz",
    },
    "saving step": {
        "thermo": 500,
        "trajectory": 500,
        "events": 2000,
        "states": 2000,
        "restart": 2000,
    },
    "states folder": "PID/states",
    "probabilities": {
        "position": 0.40,
        "volume": 0.30,
        "creation": 0.15,
        "destruction": 0.15,
    },
    "continue": True,
    "cutoff": 6.0,
    "chemical potential": "UNK",
    "seed": None,
    "max attempts": {"creation": 1, "destruction": 1},
    "max displacement": 0.175,
    "max volume change": 0.01,
}

PID_CONF = {
    "n target": "from input",
    "bonds": 30,
    "MC steps": 1,
    "dropout": 0.5,
    "compressibility scale": 10.0,
    "output file": "chem_potential.csv",
    "saving step": 5,
    "early stop": {
        "window": 5000,
        "target atoms mean": 1e-2,
        "target atoms variance": 1.5,
        "chemical potential variance": 1e-3,
    },
    "chemical potential": "UNK",
}

VOLTAGE_CONF = {
    "system": "UNK",
    "mace_model": "UNK",
    "optimizer": {
        "type": "FIRE2",
        "fmax": 0.05,
        "max steps": 5000,
    },
    "output files": {
        "thermo": "VP/thermo.csv",
        "trajectory": "VP/trj.xyz",
        "convex hull": "VP/hull.csv",
        "voltage": "VP/voltage.csv",
    },
    "states folder": "VP/states",
    "working ion": {
        "charge carried": 1,
        "chemical potential": "UNK",
    },
}

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={0:s}
#SBATCH --output={1:s}/simulation.out
#SBATCH --error={1:s}/simulation.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH -A IscrB_NEXT-LIB
#SBATCH -p boost_usr_prod

module purge
module load profile/chem-phys
module load quantum-espresso/7.2--openmpi--4.1.4--nvhpc--23.1-openblas-cuda-11.8

export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=${{{{SLURM_CPUS_PER_TASK}}}}
export CRAY_CUDA_MPS=1

ulimit -s unlimited

source /leonardo/home/userexternal/pdeangel/.bashrc
mamba activate /leonardo/home/userexternal/pdeangel/venv_mace

which pip
which python

# Run MC
{2:s}
"""

SLURM_PID = "run_pid.sh"
SLURM_EQ = "run_eq.sh"
SLURM_VOLT = "run_volt.sh"


# Function to update the YAML configuration file
def write_yaml(data, output_yaml):
    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def run_slurm(
    slurm_script: Path,
    dependency: int = None,
):
    """
    Submit the prepared SLURM script, optionally with a job dependency.

    Args:
        slurm_script (Path): Path to the SLURM batch script.
        dependency (Optional[int]): Job ID to depend on (optional).
    """
    try:
        # Construct the sbatch command
        sbatch_command = ["sbatch"]
        if dependency:
            sbatch_command.extend(["--dependency", f"afterok:{dependency}"])
        sbatch_command.append(str(slurm_script))

        # Run the sbatch command and capture output
        result = subprocess.run(sbatch_command, capture_output=True, text=True, check=True)

        # Parse the SLURM job ID from the output
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = int(output.split()[-1])
            print(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        else:
            print(f"Unexpected sbatch output: {output}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e.stderr}")
        raise typer.Exit(code=1)


def setup_eq(
    system_path: Path,
    working_dir: Path,
    mace_model_path: Path,
    chemical_potential: dict[str, Any],
    temperature: float,
    pressure: float,
):
    system = read(system_path)
    formula = Formula(system.get_chemical_formula())
    formula_reduce, _ = formula.reduce()
    system_new_path = working_dir / (formula.format("hill") + ".xyz")
    write(system_new_path, system)
    simulation_path = working_dir / "EQ"
    simulation_path.mkdir(exist_ok=True)
    eq_conf = EQ_CONF.copy()
    eq_conf.update(
        {
            "system": str(os.path.relpath(system_new_path, working_dir)),
            "mace_model": str(os.path.relpath(mace_model_path, working_dir)),
            "chemical potential": chemical_potential,
            "temperature": temperature,
            "pressure": pressure,
        }
    )
    eq_conf_path = working_dir / "conf_eq.yaml"
    eq_log_path = simulation_path / "equilibration.log"
    write_yaml(eq_conf, eq_conf_path)
    cmd = f"pymc run {os.path.relpath(eq_conf_path, working_dir)} --log-level INFO --device cuda --no-colorize --log-file {os.path.relpath(eq_log_path, working_dir)}"
    slurm_script = SLURM_SCRIPT_TEMPLATE.format(
        "EQ_" + formula_reduce.format("hill"), os.path.relpath(simulation_path, working_dir), cmd
    )
    with open(working_dir / SLURM_EQ, "w") as f:
        f.write(slurm_script)


def setup_volt(
    system_path: Path, working_dir: Path, mace_model_path: Path, charge_carried: int, chemical_potential: dict[str, Any]
):
    system = read(system_path)
    formula = Formula(system.get_chemical_formula())
    formula_reduce, _ = formula.reduce()
    system_new_path = working_dir / (formula.format("hill") + ".xyz")
    write(system_new_path, system)
    simulation_path = working_dir / "VP"
    simulation_path.mkdir(exist_ok=True)
    volt_conf = VOLTAGE_CONF.copy()
    volt_conf.update(
        {
            "system": str(os.path.relpath(system_new_path, working_dir)),
            "mace_model": str(os.path.relpath(mace_model_path, working_dir)),
            "working ion": {"charge carried": charge_carried, "chemical potential": chemical_potential},
        }
    )
    volt_conf_path = working_dir / "conf_volt.yaml"
    volt_log_path = simulation_path / "voltage.log"
    write_yaml(volt_conf, volt_conf_path)
    cmd = f"pymc zerok-voltage {os.path.relpath(volt_conf_path, working_dir)} --log-level INFO --device cuda --no-colorize --log-file {os.path.relpath(volt_log_path, working_dir)}"
    slurm_script = SLURM_SCRIPT_TEMPLATE.format(
        "VP_" + formula_reduce.format("hill"), os.path.relpath(simulation_path, working_dir), cmd
    )
    with open(working_dir / SLURM_VOLT, "w") as f:
        f.write(slurm_script)


def setup_pid(
    system_path: Path,
    working_dir: Path,
    root_working_dir: Path,
    target_atoms: int,
    mace_model_path: Path,
    chemical_potential: dict[str, Any],
    temperature: float,
    pressure: float,
    mc_steps: int = int(0.5e6),
):
    # Equilibration MC setup
    # eq_dir = working_dir / "EQ"
    # eq_dir.mkdir(exist_ok=True)
    # states_dir = eq_dir / "states"
    # states_dir.mkdir(exist_ok=True)
    system = read(system_path)
    formula = Formula(system.get_chemical_formula())
    formula_reduce, _ = formula.reduce()
    simulation_path = working_dir / "PID"
    simulation_path.mkdir(exist_ok=True)
    pid_mc_conf = PID_MC_CONF.copy()
    root_working_dir = Path(root_working_dir)
    pid_mc_conf.update(
        {
            "steps": int(mc_steps),
            "system": str(os.path.relpath(root_working_dir / "EQ" / "restart.xyz", working_dir)),
            "mace_model": str(os.path.relpath(mace_model_path, working_dir)),
            "chemical potential": chemical_potential,
            "temperature": temperature,
            "pressure": pressure,
        }
    )
    mc_conf_path = working_dir / "conf_mc.yaml"
    write_yaml(pid_mc_conf, mc_conf_path)
    #
    pid_conf = PID_CONF.copy()
    pid_conf.update(
        {
            "system": str(os.path.relpath(root_working_dir / "EQ" / "restart.xyz", working_dir)),
            "mace_model": str(os.path.relpath(mace_model_path, working_dir)),
            "n target": int(target_atoms),
            "bonds": 15,
            "compressibility scale": 15.0,
            "output file": "chem_potential.csv",
            "saving step": 5,
            "early stop": {
                "window": 10000,
                "target atoms mean": 1e-3,
                "target atoms variance": 1.5,
                "chemical potential variance": 1e-5,
            },
            "chemical potential": chemical_potential,
        }
    )
    pid_conf_path = working_dir / "conf_pid.yaml"
    write_yaml(pid_conf, pid_conf_path)
    #
    pid_log_path = simulation_path / "pid_chemical_potential.log"

    cmd = f"pymc pid {os.path.relpath(mc_conf_path, working_dir)} {os.path.relpath(pid_conf_path, working_dir)} --log-level DEBUG --device cuda --no-colorize --log-file {os.path.relpath(pid_log_path, working_dir)}"
    slurm_script = SLURM_SCRIPT_TEMPLATE.format(
        f"PID_{target_atoms:03d}_" + formula_reduce.format("hill"), os.path.relpath(simulation_path, working_dir), cmd
    )
    with open(working_dir / SLURM_PID, "w") as f:
        f.write(slurm_script)


# Prepare function
@app.command(help="Prepare input files and SLURM scripts.")
def prepare(
    system_path: Annotated[Path, typer.Argument(help="Path to the system CIF/XYZ file")],
    pure_ion_path: Annotated[Path, typer.Argument(help="Path to the pure ion configuration file")],
    mace_model_path: Annotated[Path, typer.Argument(help="Path to the MACE model file")],
    temperature: Annotated[float, typer.Option(help="Ensemble Temperature (K)")] = 300.0,
    pressure: Annotated[float, typer.Option(help="Ensemble Pressure (bar)")] = 1.0,
    working_dir: Annotated[Path, typer.Option(help="Path to the working directory")] = None,
    charge_carried: Annotated[int, typer.Option(help="Charge carried by the ion")] = 1,
    pid_max_steps: Annotated[float, typer.Option(help="Max MC steps in PID simulation")] = 0.5e6,
    max_ion_fraction: Annotated[float, typer.Option(help="Max ion fraction in formula unit")] = 1.0,
    min_ion_fraction: Annotated[float, typer.Option(help="Min ion fraction in formula unit")] = 0.0,
):
    """
    Set up SLURM scripts and input files for simulations.
    """
    system_path = Path(system_path).resolve()
    system = read(system_path)
    mace_model_path = Path(mace_model_path)
    pure_ion_path = Path(pure_ion_path).resolve()
    pure_ion = read(pure_ion_path)
    if working_dir is None:
        model_name = os.path.basename(mace_model_path).split(".")[0]
        working_dir = f"./{system.get_chemical_formula()}_{temperature:.1f}K_{pressure:.1f}bar_{model_name}"
    working_dir = Path(working_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    ion_element = pure_ion.get_chemical_symbols()[0]
    if not np.all(np.array(pure_ion.get_chemical_symbols()) == ion_element):
        raise ValueError("the input pure_ion_path should be pure element system")
    # EQ
    chemical_potential = {
        ion_element: str(os.path.relpath(pure_ion_path, working_dir)),
    }
    setup_eq(system_path, working_dir, mace_model_path, chemical_potential, temperature, pressure)
    # VP
    setup_volt(system_path, working_dir, mace_model_path, charge_carried, chemical_potential)
    # PID
    n_ions_max = len(np.where(np.array(system.get_chemical_symbols()) == ion_element)[0])
    chemical_potential_pid = {
        ion_element: 5.0,
    }
    for i, n_ions in enumerate(
        range(int(np.floor(n_ions_max * min_ion_fraction)), int(np.floor(n_ions_max * max_ion_fraction)))
    ):
        pid_working_dir = working_dir / f"{i:03d}-n_ions-{n_ions:d}"
        pid_working_dir.mkdir(exist_ok=True)
        setup_pid(
            system_path,
            pid_working_dir,
            working_dir,
            n_ions,
            mace_model_path,
            chemical_potential_pid,
            temperature,
            pressure,
            mc_steps=int(pid_max_steps),
        )

    typer.echo(f"Preparation completed. Files are in {working_dir}")


# Run function
@app.command(help="Run simulations using SLURM.")
def run(
    working_dir: Annotated[Path, typer.Argument(help="Path to the working directory")],
):
    """
    Submit the prepared SLURM script.
    """
    script_dir = Path(".").resolve()
    # run EQ
    os.chdir(working_dir)
    try:
        job_id_eq = run_slurm(SLURM_EQ)
        typer.echo("Submitted equilibration simulation")
    except:  # noqa:E722
        typer.echo("Problem submitting equilibration simulation")
        raise typer.Exit(code=1)
    # run VOLT
    try:
        job_id_volt = run_slurm(SLURM_VOLT)  # noqa:F841
        typer.echo("Submitted 0-K voltage simulation")
    except:  # noqa:E722
        typer.echo("Problem submitting  0-K voltage simulation")
        raise typer.Exit(code=1)
    os.chdir(script_dir)
    conf_dirs = glob.glob(str(working_dir / "*n_ions*"))
    for dir_ in conf_dirs:
        try:
            os.chdir(dir_)
            job_id_pid = run_slurm(SLURM_PID, job_id_eq)  # noqa:F841
            os.chdir(script_dir)
            typer.echo(f"Submitted pid simulation depending on job {job_id_eq} ({dir_})")
        except:  # noqa:E722
            typer.echo(f"Problem submitting pid simulation ({dir_})")
            os.chdir(script_dir)
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
