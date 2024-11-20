import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_masses, atomic_numbers
from ase.data.vdw_alvarez import vdw_radii
from ase.geometry import get_distances, wrap_positions
from ase.neighborlist import NeighborList
from ase.units import kB
from loguru import logger
from numpy.random import Generator
from scipy.constants import N_A, Planck, elementary_charge

from mc_mace.utils.neighborlist import FastPrimitiveNeighborList
from mc_mace.utils.profiler import MethodProfiler

ATTEMPT_TYPE = ["volume", "position", "creation", "destruction"]
vdw_radii = np.nan_to_num(vdw_radii, nan=np.nanmean(vdw_radii))

profiler_main = MethodProfiler(name="Profiling MC steps")
profiler_sub = MethodProfiler(name="Profiling Sub-routine")


class MC:
    """
    A Monte Carlo (MC) simulation class for atomic systems.

    This class provides Metropolis MC simulations for volume changes,
    atomic position changes, atom creation, and destruction.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator: Calculator,
        mus: list[float],
        insert_elements: list[str],
        T: float,
        P: float,
        steps: int,
        random_number_gen: None | Generator = None,
        cutoff: float = 6.0,
        max_displacement: float = 0.1,
        max_volume_change: float = 0.0,
    ):
        """
        Initialize the Metropolis MC simulation.

        Args:
            atoms (Atoms): ASE Atoms object representing the system.
            calculator (Calculator): ASE-compatible calculator for energy calculations.
            mus (list[float]): Chemical potentials for each element.
            insert_elements (list[str]): Elements allowed for insertion.
            T (float): Temperature in Kelvin.
            P (float): Pressure in eV/Å³.
            steps (int): Number of MC steps.
            random_number_gen (np.random.Generator): Optional random number generator.
            cutoff (float): Neighbor list cutoff radius (default: 6.0 Å).
            max_displacement (float): Maximum atomic displacement per move.
            max_volume_change (float): Maximum volume change per step.
        """
        self.atoms_old = atoms.copy()
        self.atoms_new = atoms.copy()
        self.mus = mus
        self.insert_elements = insert_elements
        self.T = T
        self.beta = 1 / (kB * T)
        self.P = P
        self.max_step = {"volume": max_volume_change, "position": max_displacement}
        self.steps = steps
        self._i_step = 0
        self.cutoff = cutoff
        self.calculator = calculator
        self.atoms_old.set_calculator(calculator)
        self.atoms_new.set_calculator(calculator)
        self.neighbor_list_old = self.build_neighbor_list(self.atoms_old)
        self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
        self.rng = random_number_gen or np.random.default_rng(seed=-1)
        self.accept = {t: 0 for t in ATTEMPT_TYPE}
        self.reject = {t: 0 for t in ATTEMPT_TYPE}
        self._creation_attempts = 100
        self._destruction_attempts = 100
        self._aindex = 0
        self.deBroglie_wl: list[float] = [0.0]
        self._E_old = 0.0
        self._E_new = 0.0
        self._compute_deBroglie_wavelength()

    ### Private Methods ###

    def _compute_deBroglie_wavelength(self) -> None:
        """
        Compute the thermal de Broglie wavelength for each insertable element.
        """
        eV2J = elementary_charge
        au2kg = 1e-3 / N_A
        m2A = 1e10
        self.deBroglie_wl = [
            np.sqrt(Planck**2 / (2 * np.pi * (atomic_masses[atomic_numbers[el]] * au2kg) * kB * eV2J * self.T)) * m2A
            for el in self.insert_elements
        ]

    def _get_next_moving_atom(self) -> int:
        """
        Get the next atom index for movement.

        Returns:
            int: The index of the next atom to be moved.
        """
        self._aindex += 1
        return self._aindex % len(self.atoms_old)

    def _update_step(self) -> None:
        """
        Increment the step counter for the simulation.
        """
        self._i_step += 1

    def _logger_prefix(self) -> str:
        """
        Generate a standardized logging prefix for the current step.

        Returns:
            str: A formatted prefix string indicating the current step.
        """
        return "[" + f"Step {self._i_step:d}".center(16, " ") + f"| {self._i_step/(self.steps)*100:>6.2f} % " + "] "

    ### State Management Methods ###

    @profiler_sub.track
    def build_neighbor_list(self, atoms: Atoms) -> NeighborList:
        """
        Build a neighbor list for the given atomic configuration.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            NeighborList: A neighbor list object for the atoms.
        """
        neighbor_list = NeighborList(
            self.cutoff,
            self_interaction=False,
            bothways=True,
            primitive=FastPrimitiveNeighborList,
        )
        neighbor_list.update(atoms)
        return neighbor_list

    @profiler_sub.track
    def get_state_configuration(self) -> Atoms:
        """
        Get a copy of the current atomic configuration.

        Returns:
            Atoms: Copy of the current configuration.
        """
        return self.atoms_old.copy()

    @profiler_sub.track
    def get_state_energy(self) -> float:
        """
        Get the potential energy of the current state.

        Returns:
            float: Potential energy in eV.
        """
        return self._E_old

    @profiler_sub.track
    def get_state_volume(self) -> float:
        """
        Get the volume of the current state.

        Returns:
            float: Volume in Å³.
        """
        return float(self.atoms_old.get_volume())

    @profiler_sub.track
    def get_energy_difference(self) -> float:
        """
        Calculate the energy difference between the trial and current states.

        Returns:
            float: Energy difference in eV.
        """
        self._E_old = self.atoms_old.get_potential_energy()
        self._E_new = self.atoms_new.get_potential_energy()
        logger.debug(self._logger_prefix() + f"Energy difference: E_old={self._E_old}, E_new={self._E_new}")
        return self._E_new - self._E_old

    @profiler_sub.track
    def restore_state(self) -> None:
        """
        Restore the trial state to match the current state.
        """
        self.atoms_new = self.atoms_old.copy()
        self.atoms_new.set_calculator(self.calculator)

    @profiler_sub.track
    def update_state(self) -> None:
        """
        Update the current state to match the trial state.
        """
        self.atoms_old = self.atoms_new.copy()
        self.atoms_old.set_calculator(self.calculator)
        self._E_old = self._E_new
        self.atoms_new = self.atoms_new.copy()
        self.atoms_new.set_calculator(self.calculator)

    @profiler_sub.track
    def update_neighbor_list(self, which: str = "all") -> None:
        """
        Update the neighbor list for the system.

        Args:
            which (str): Which neighbor list to update ('old', 'new', or 'all').
        """
        if which in ["all", "old"]:
            try:
                self.neighbor_list_old.update(self.atoms_old)
                logger.debug(self._logger_prefix() + "Updated `old` state neighbor list")
            except ValueError:
                self.neighbor_list_old = self.build_neighbor_list(self.atoms_old)
                logger.debug(self._logger_prefix() + "Updated `new` state neighbor list")

        if which in ["all", "new"]:
            try:
                self.neighbor_list_new.update(self.atoms_new)
            except ValueError:
                self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)

    ### Utility Methods ###

    @profiler_sub.track
    def wrap_position(self, atoms: Atoms) -> Atoms:
        """
        Wrap atomic positions into the periodic boundary conditions.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            Atoms: Atoms object with wrapped positions.
        """
        atoms.set_positions(wrap_positions(atoms.get_positions(), atoms.cell, pbc=True))
        return atoms

    @profiler_sub.track
    def get_vdw_radii(self, atoms: Atoms) -> np.ndarray:
        """
        Get van der Waals radii for all atoms in the system.

        Args:
            atoms (Atoms): ASE Atoms object.

        Returns:
            np.ndarray: Array of van der Waals radii.
        """
        return np.array([vdw_radii[ai] for ai in atoms.get_atomic_numbers()])

    @profiler_sub.track
    def get_min_distance(self, index: int) -> float:
        """
        Get the minimum distance of an atom to its neighbors.

        Args:
            index (int): Index of the atom.

        Returns:
            float: Minimum distance in Å.
        """
        ngh_indices, offsets = self.neighbor_list_new.get_neighbors(index)
        ngh_indices = np.unique(ngh_indices[ngh_indices != index])

        if len(ngh_indices) == 0:
            return self.cutoff  # No neighbors, no overlap

        neighbor_positions = self.atoms_new.positions[ngh_indices]
        _, distances = get_distances(
            self.atoms_new.positions[index],
            p2=neighbor_positions,
            cell=self.atoms_new.cell,
            pbc=True,
        )
        return float(np.min(distances))

    @profiler_sub.track
    def check_overlap(self, index: int) -> bool:
        """
        Check for overlaps of the atom at the given index with its neighbors.

        Args:
            index (int): Index of the atom.

        Returns:
            bool: True if overlap is detected, False otherwise.
        """
        ngh_indices, offsets = self.neighbor_list_new.get_neighbors(index)
        # Fixing Vesin Neighbors bug
        ngh_indices = np.unique(ngh_indices)
        ngh_indices = ngh_indices[ngh_indices != index]

        if len(ngh_indices) == 0:
            overlap = False  # No neighbors, no overlap
            logger.debug(self._logger_prefix() + f"No overlap for new atom {index} position (No neighbors)")
        else:
            # Vectorized position difference with periodic boundary conditions
            neighbor_positions = self.atoms_new.positions[ngh_indices]  # + offsets @ self.atoms_new.cell
            # position_diff = neighbor_positions - self.atoms_new.positions[index]
            _, distances = get_distances(
                self.atoms_new.positions[index],
                p2=neighbor_positions,
                cell=self.atoms_new.cell,
                pbc=True,
            )
            # distances = np.linalg.norm(position_diff, axis=1)

            vdw_radii = self.get_vdw_radii(self.atoms_new)
            # Calculate the vdW threshold distances
            overlap_thresholds = (vdw_radii[index] + vdw_radii[ngh_indices]) * 0.45

            # Vectorized overlap check
            overlap = np.any(distances < overlap_thresholds)
            if overlap:
                i_overlap = np.where(distances < overlap_thresholds)[1]
                n_overlap = len(i_overlap)
                logger.debug(self._logger_prefix() + f"Found {n_overlap} overlap for new atom {index} position")
                for i in i_overlap:
                    logger.debug(
                        self._logger_prefix()
                        + f"\tOverlap with atom {ngh_indices[i]}: distance={distances[0,i]} A, vdw={overlap_thresholds[i]}A"
                    )
            else:
                logger.debug(
                    self._logger_prefix()
                    + f"No overlap for new atom {index} position (min distance: {np.min(distances)} A)"
                )
        return overlap

    @profiler_sub.track
    def tune_max_steps(self) -> None:
        """
        Dynamically tune the maximum step sizes for volume and position changes
        based on the acceptance rate.

        The method adjusts step sizes to maintain an acceptance rate between 45% and 55%.
        A higher acceptance rate results in increased step sizes, while a lower
        acceptance rate decreases the step sizes.

        For each attempt type (volume and position):
        - If acceptance rate > 55%, step size increases by 5%.
        - If acceptance rate < 45%, step size decreases by 5%.
        """
        for step_type in ["volume", "position"]:
            if self.accept[step_type] is None:
                continue
            logger.info(self._logger_prefix() + f"Tuning maximum step sizes for {step_type} attempt types.")
            total_attempts = self.accept[step_type] + self.reject[step_type]

            if total_attempts < 100:
                logger.warning(
                    self._logger_prefix()
                    + f"Not enough attempts for tuning '{step_type}' (only {total_attempts} attempts)."
                )
                continue

            # Calculate acceptance rate
            acceptance_rate = self.accept[step_type] / total_attempts
            logger.debug(
                self._logger_prefix()
                + f"'{step_type}' acceptance rate = {self.accept[step_type]} / {total_attempts} = {acceptance_rate:.2%}"
            )

            # Adjust step sizes
            if acceptance_rate > 0.55:
                self.max_step[step_type] *= 1.05
                logger.debug(self._logger_prefix() + f"Increasing '{step_type}' step size by 5%.")
            elif acceptance_rate < 0.45:
                self.max_step[step_type] *= 0.95
                logger.debug(self._logger_prefix() + f"Decreasing '{step_type}' step size by 5%.")

            # Reset counters for the next tuning interval
            self.accept[step_type] = 0
            self.reject[step_type] = 0

            logger.info(
                self._logger_prefix() + f"New maximum step size for '{step_type}': {self.max_step[step_type]:.4f}"
            )

    ### MC Step Methods ###
    @profiler_main.track
    def attempt_nothing(self) -> bool:
        """Attempt to nothing."""
        logger.info(self._logger_prefix() + "Attempt nothing (compute configuration energy)")
        delta_E = self.get_energy_difference()
        logger.debug(self._logger_prefix() + f"PE difference, ΔE={delta_E:.3f} eV")
        P_acc = np.min([1, np.exp(-self.beta * (delta_E))])
        logger.debug(self._logger_prefix() + f"Acceptance probability, P_acc={P_acc:.3e}")
        logger.info(
            self._logger_prefix() + f"PE difference ΔE={delta_E:.3f} eV, Acceptance probability, P_acc={P_acc:.3e}"
        )
        x = self.rng.random()
        logger.debug(self._logger_prefix() + f"Random number x={x:.3f}")
        acc_state = bool(x < P_acc)
        logger.debug(self._logger_prefix() + f"Accepted={acc_state}")
        if acc_state:
            logger.success(f"Accepted (E = {self._E_new:.3f})")
        else:
            self.restore_state()
            logger.info(self._logger_prefix() + f"Rejected (E = {self._E_old:.3f})")
        # Moving to next step:
        self._update_step()
        return acc_state

    @profiler_main.track
    def attempt_volume_change(self) -> bool:
        """Attempt to change the volume of the system."""
        V_old = self.atoms_old.get_volume()
        max_volume_change = self.max_step["volume"]
        delta_V = self.rng.uniform(-max_volume_change, max_volume_change)
        V_new = V_old + delta_V
        logger.info(
            self._logger_prefix()
            + f"Attempt volume change {V_old:>6.3g} A^3 -> {V_new:>6.3g} A^3 (DeltaV={delta_V:>6.3g} A^3)"
        )
        scale_factor = (V_new / V_old) ** (1 / 3)
        self.atoms_new.set_cell(self.atoms_new.cell * scale_factor, scale_atoms=True)
        length = self.atoms_old.cell.lengths()
        logger.debug(self._logger_prefix() + f"Cell (old) length=[{length[0]}, {length[1]}, {length[2]}]")
        length = self.atoms_new.cell.lengths()
        logger.debug(self._logger_prefix() + f"Cell (new) length=[{length[0]}, {length[1]}, {length[2]}]")
        Na = len(self.atoms_old)
        delta_E = self.get_energy_difference()
        logger.debug(self._logger_prefix() + f"PE difference, ΔE={delta_E:.3f} eV")
        P_acc = np.min(
            [
                1,
                np.exp(-self.beta * (delta_E + self.P * (V_new - V_old)))
                / np.exp(-Na * np.log(V_new / V_old)),  # https://doi.org/10.1103/PhysRevE.103.L061303 Eq.(7)
            ]
        )
        logger.debug(self._logger_prefix() + f"Acceptance probability, P_acc={P_acc:.3e}")
        logger.info(
            self._logger_prefix() + f"PE difference ΔE={delta_E:.3f} eV, Acceptance probability, P_acc={P_acc:.3e}"
        )
        x = self.rng.random()
        logger.debug(self._logger_prefix() + f"Random number x={x:.3f}")
        acc_state = bool(x < P_acc)
        logger.debug(self._logger_prefix() + f"Accepted={acc_state}")
        if acc_state:
            logger.success(self._logger_prefix() + f"Accepted (E = {self._E_new:.3f})")
            self.accept["volume"] += 1
        else:
            self.reject["volume"] += 1
            self.restore_state()
            logger.info(self._logger_prefix() + f"Rejected (E = {self._E_old:.3f})")
        # Moving to next step:
        self._update_step()
        return acc_state

    @profiler_main.track
    def attempt_position_change(self) -> bool:
        """Attempt to move an atom in the system."""
        index = self._get_next_moving_atom()
        old_position = np.array(self.atoms_old.positions[index])
        max_displacement = self.max_step["position"]
        displacement = self.rng.uniform(-max_displacement, max_displacement, size=(3,))
        logger.info(
            self._logger_prefix()
            + f"Attempt moving atom {index:>5d} ({self.atoms_old.get_chemical_symbols()[index]:3s}) by a displacement [{displacement[0]:6.3f}, {displacement[1]:6.3f}, {displacement[2]:6.3f}]"
        )
        new_position = old_position + displacement
        self.atoms_new.positions[index] = new_position
        self.atoms_new = self.wrap_position(self.atoms_new)
        new_position = self.atoms_new.positions[index]
        self.update_neighbor_list("new")
        logger.debug(
            self._logger_prefix()
            + f"Attempt position change [{old_position[0]:.3f}, {old_position[1]:.3f}, {old_position[2]:.3f}] -> [{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}]"
        )

        if not self.check_overlap(index):
            delta_E = self.get_energy_difference()
            logger.debug(self._logger_prefix() + f"PE difference, ΔE={delta_E:.3f} eV")
            P_acc = np.min(
                [
                    1,
                    np.exp(-self.beta * delta_E),  # https://doi.org/10.1103/PhysRevE.103.L061303 Eq.(7)
                ]
            )
            logger.debug(self._logger_prefix() + f"Acceptance probability, P_acc={P_acc:.3e}")
            logger.info(
                self._logger_prefix() + f"PE difference ΔE={delta_E:.3f} eV, Acceptance probability, P_acc={P_acc:.3e}"
            )
            x = self.rng.random()
            logger.debug(self._logger_prefix() + f"Random number x={x:.3f}")
            acc_state = bool(x < P_acc)
            logger.debug(self._logger_prefix() + f"Accepted={acc_state}")
        else:
            logger.debug(self._logger_prefix() + f"Atoms {index} overlaps")
            acc_state = False
            logger.debug(self._logger_prefix() + f"Accepted={acc_state}")
        if acc_state:
            logger.success(self._logger_prefix() + f"Accepted (E = {self._E_new:.3f})")
            self.accept["position"] += 1
        else:
            self.reject["position"] += 1
            self.restore_state()
            logger.info(self._logger_prefix() + f"Rejected (E = {self._E_old:.3f})")
        # Moving to next step:
        self._update_step()
        return acc_state

    @profiler_main.track
    def attempt_creation(self) -> bool:
        """Attempt to create a new atom in the system."""
        i_element = self.rng.choice(len(self.insert_elements))
        mu = self.mus[i_element]
        element = self.insert_elements[i_element]
        deBroglie_wl = self.deBroglie_wl[i_element]
        Na = len(self.atoms_old)
        V = self.atoms_old.get_volume()
        k = 0
        acc_state = False
        logger.info(self._logger_prefix() + f"Attempt creation of {element} atom")
        logger.info(self._logger_prefix() + f"Running {self._creation_attempts} attempts of creations;")
        while k < self._creation_attempts and acc_state is False:
            k += 1
            log_second_prefix = f"[ k={k:>3d} / {self._creation_attempts:>3d} ] "
            new_r_fractional = self.rng.uniform(0, 1, size=3)
            new_r_cartesian = self.atoms_old.cell.cartesian_positions(new_r_fractional)
            logger.debug(
                self._logger_prefix()
                + log_second_prefix
                + f"Attempt to add {element} atom in position [{float(new_r_cartesian[0]):.3f}, {float(new_r_cartesian[1]):.3f}, {float(new_r_cartesian[2]):.3f}] ([{float(new_r_fractional[0]):.3f}, {float(new_r_fractional[1]):.3f}, {float(new_r_fractional[2]):.3f}])"
            )
            self.atoms_new += Atoms(element, positions=[new_r_cartesian])
            self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
            logger.debug(self._logger_prefix() + log_second_prefix + f"New atom = {self.atoms_new[Na]}")
            if not self.check_overlap(len(self.atoms_new) - 1):
                delta_E = self.get_energy_difference()
                min_dist = self.get_min_distance(len(self.atoms_new) - 1)
                logger.debug(
                    self._logger_prefix() + log_second_prefix + f"Distance from the nearest neighbor = {min_dist} A"
                )
                logger.debug(
                    self._logger_prefix()
                    + log_second_prefix
                    + f"PE difference, ΔE={delta_E} eV (DeltaE-mu={delta_E-mu} eV)"
                )
                Wn = 1 / k  # Rosenbluth factor (hard sphere)
                P_acc = np.min(
                    [
                        1,
                        np.exp(-self.beta * (delta_E - mu))
                        * V
                        / (deBroglie_wl**3 * (Na + 1))
                        * Wn,  # https://doi.org/10.1103/PhysRevE.103.L061303 Eq.(5)
                    ]
                )
                logger.debug(self._logger_prefix() + log_second_prefix + f"Acceptance probability, P_acc={P_acc:.3e}")
                x = self.rng.random()
                logger.debug(self._logger_prefix() + log_second_prefix + f"Random number x={x:.3}")
                acc_state = bool(x < P_acc)
                logger.debug(self._logger_prefix() + log_second_prefix + f"Accepted={acc_state}")
            else:
                logger.debug(self._logger_prefix() + log_second_prefix + "Inserted atom overlaps")
                acc_state = False
                logger.debug(self._logger_prefix() + log_second_prefix + f"Accepted={acc_state}")
            if acc_state is False:
                self.restore_state()
                # self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
        if acc_state:
            logger.info(
                self._logger_prefix()
                + f"PE difference, ΔE = {delta_E:.3f} eV (ΔE-μ = {delta_E-mu:.3f} eV), Acceptance probability, P_acc={P_acc:.3e} (Wn={Wn:.3f})"
            )
            logger.success(self._logger_prefix() + f"Accepted (E = {self._E_new:.3f})")
            self.accept["creation"] += 1
        else:
            self.reject["creation"] += 1
            self.restore_state()
            self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
            logger.info(self._logger_prefix() + f"Rejected (E = {self._E_old:.3f})")
        # Moving to next step:
        self._update_step()
        return acc_state

    @profiler_main.track
    def attempt_destruction(self) -> bool:
        """Attempt to remove an atom from the system."""
        i_element = self.rng.choice(len(self.insert_elements))
        mu = self.mus[i_element]
        element = self.insert_elements[i_element]
        deBroglie_wl = self.deBroglie_wl[i_element]
        Na = len(self.atoms_old)
        V = self.atoms_old.get_volume()
        k = 0
        acc_state = False
        atoms_candidates = np.where(self.atoms_old.get_atomic_numbers() == atomic_numbers[element])[0]
        self.rng.shuffle(atoms_candidates)
        logger.info(self._logger_prefix() + f"Attempt destruction of {element} atom")
        logger.info(self._logger_prefix() + f"Running {self._creation_attempts} attempts of destruction")
        if len(atoms_candidates) == 0:
            logger.warning(self._logger_prefix() + f"No {element} atoms in the system")
            acc_state = False
        for index in atoms_candidates:
            if len(self.atoms_new) > 1:
                k += 1
                log_second_prefix = f"[ k={k:>3d} / {self._creation_attempts:>3d} ] "
                position = self.atoms_new.positions[index, :]
                position_frac = self.atoms_old.cell.scaled_positions(position)
                logger.debug(
                    self._logger_prefix()
                    + log_second_prefix
                    + f"Attempt to remove {element} (idx:{index}) atom in position [{float(position[0]):.3f}, {float(position[1]):.3f}, {float(position[2]):.3f}] ([{float(position_frac[0]):.3f}, {float(position_frac[1]):.3f}, {float(position_frac[2]):.3f}])"
                )
                logger.debug(self._logger_prefix() + log_second_prefix + f"removing atom = {self.atoms_new[index]}")
                del self.atoms_new[index]
                delta_E = self.get_energy_difference()
                logger.debug(
                    self._logger_prefix()
                    + log_second_prefix
                    + f"PE difference, ΔE = {delta_E:.3f} eV (ΔE+μ ={delta_E+mu:.3f} eV)"
                )
                Wn = k / 1  # Rosenbluth factor (hard sphere)
                P_acc = np.min(
                    [
                        1,
                        np.exp(-self.beta * (delta_E + mu))
                        * deBroglie_wl**3
                        * Na
                        / V
                        * Wn,  # https://doi.org/10.1103/PhysRevE.103.L061303 Eq.(5)
                    ]
                )
                logger.debug(self._logger_prefix() + log_second_prefix + f"Acceptance probability, P_acc={P_acc:.3e}")
                x = self.rng.random()
                logger.debug(self._logger_prefix() + log_second_prefix + f"Random number x={x:.3f}")
                acc_state = x < P_acc
                logger.debug(self._logger_prefix() + log_second_prefix + f"Accepted={acc_state}")
            else:
                logger.warning(
                    self._logger_prefix() + log_second_prefix + "On one atoms in the box, impossible removing atoms"
                )
                acc_state = False
                logger.debug(self._logger_prefix() + log_second_prefix + f"Accepted={acc_state}")
            if acc_state is False:
                self.restore_state()
            if acc_state is True or k >= self._destruction_attempts:
                break
                # self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
        if acc_state:
            logger.info(
                self._logger_prefix()
                + f"PE difference, ΔE = {delta_E:.3f} eV (ΔE+μ = {delta_E+mu:.3f} eV), Acceptance probability, P_acc={P_acc:.3e} (Wn={Wn:.3f})"
            )
            logger.success(self._logger_prefix() + f"Accepted (E = {self._E_new:.3f})")
            self.accept["destruction"] += 1
        else:
            self.reject["destruction"] += 1
            self.restore_state()
            self.neighbor_list_new = self.build_neighbor_list(self.atoms_new)
            logger.info(self._logger_prefix() + f"Rejected (E = {self._E_old:.3f})")
        # Moving to next step:
        self._update_step()
        return acc_state

    def ended(self) -> bool:
        """Check if the simulation finished"""
        return self._i_step > self.steps

    ### Reporting ###

    def print_report(self) -> None:
        """
        Print the profiling report for MC steps and sub-routines.
        """
        for line in profiler_main.report():
            logger.info(line)
        logger.info("")
        for line in profiler_sub.report():
            logger.info(line)
