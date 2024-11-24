from pathlib import Path

import numpy as np
from loguru import logger
from numpy.random import Generator

from mc_mace.mc import MC
from mc_mace.utils.profiler import MCProfiler

from .abc_ensemble import Ensemble
from .exceptions import InvalidEnsembleAttemptType

mc_profiler = MCProfiler()


class NPT(Ensemble):
    """
    NPT Monte Carlo Ensemble.

    This class implements the NPT ensemble, where the simulation is conducted under
    constant pressure and temperature. It extends the `Ensemble` base class and defines
    methods for move selection, execution, and simulation reporting.

    Attributes:
        step_probability (dict[str, float]): Probability distribution for different move types.
        allowed_steps (list[str]): List of allowed move types in the simulation.
        _p (np.ndarray): Normalized probabilities for move selection.
    """

    def __init__(
        self,
        engine: MC,
        steps: int,
        step_probability: dict[str, float] = {"position": 0.5, "volume": 0.5},
        random_number_gen: Generator | None = None,
        out_thermo: str | Path | None = None,
        out_trj: str | Path | None = None,
        out_events: str | Path | None = None,
        out_state_folder: str | Path | None = None,
        out_restart: str | Path | None = None,
        save_trj_step: int | None = None,
        save_thermo_step: int | None = None,
        save_events_step: int | None = None,
        save_state_step: int | None = None,
        save_restart_step: int | None = None,
        tunning_step: int | None = None,
    ) -> None:
        """
        Initialize the NPT ensemble simulation.

        Args:
            engine (MC): The Monte Carlo engine managing the simulation state and moves.
            steps (int): Total number of Monte Carlo steps in the simulation.
            step_probability (dict[str, float]): Probability distribution for different move types
                (default: {"position": 0.5, "volume": 0.5}).
            random_number_gen (Generator | None): Random number generator instance (default: numpy RNG).
            out_thermo (str | Path | None): Path for thermodynamic output file.
            out_trj (str | Path | None): Path for trajectory output file.
            out_events (str | Path | None): Path for events output file.
            out_state_folder (str | Path | None): Folder path for saving state files.
            out_restart (str | Path | None): Path for restart file output.
            save_trj_step (int | None): Frequency for saving trajectory data.
            save_thermo_step (int | None): Frequency for saving thermodynamic data.
            save_events_step (int | None): Frequency for saving event data.
            save_state_step (int | None): Frequency for saving state data.
            save_restart_step (int | None): Frequency for saving restart files.
            tunning_step (int | None): Frequency for tuning simulation parameters.
        """
        super().__init__(
            engine=engine,
            steps=steps,
            random_number_gen=random_number_gen,
            out_thermo=out_thermo,
            out_trj=out_trj,
            out_events=out_events,
            out_restart=out_restart,
            out_state_folder=out_state_folder,
            save_trj_step=save_trj_step,
            save_thermo_step=save_thermo_step,
            save_events_step=save_events_step,
            save_state_step=save_state_step,
            save_restart_step=save_restart_step,
            tunning_step=tunning_step,
        )
        self.step_probability = step_probability
        self.allowed_steps = ["position", "volume"]
        self._p = np.array([self.step_probability[k] for k in self.allowed_steps])
        if np.abs(self._p.sum() - 1.0) > 1e-8:
            logger.warning("The probability for the MC steps do not sum up to 1. We are going to normalize it")
            self._p = self._p / self._p.sum()

    def start_msg(self) -> None:
        """
        Print simulation initialization message.

        This method logs the details of the simulation setup, including initial atom count,
        step probabilities, and output file configuration.
        """
        logger.info(" NPT ".center(120, "="))
        logger.info("")
        logger.info(
            f"Stating NPT-MC simulation (N={len(self.engine.atoms_old)}, P={self.engine.P:8.5g} eV/A, T={self.engine.T:8.2g} K)".center(
                120, " "
            )
        )
        logger.info(f"MC steps = {self.steps}")
        logger.info(
            f"Position max step = {self.engine.max_step['position']:>10.3g} A (probability={self.step_probability['position']})"
        )
        logger.info(
            f"Volume max step = {self.engine.max_step['volume']:>10.3g} A^3 (probability={self.step_probability['volume']})"
        )
        logger.info("")
        if self.tunning_step is not None:
            logger.info(f"Tuning step every {self.tunning_step} steps")
        if self.save_trj_step:
            logger.info(f"Saving trajectory in `{self.out_trj}` every {self.save_trj_step} steps")
        if self.save_thermo_step:
            logger.info(f"Saving thermo in `{self.out_thermo}` every {self.save_thermo_step} steps")
        if self.save_events_step:
            logger.info(f"Saving events record in `{self.out_events}` every {self.save_events_step} steps")
        if self.save_state_step:
            logger.info(f"Saving chemical state in `{self.out_state_folder}` folder every {self.save_state_step} steps")
        logger.info("")

    def print_step_time(self) -> None:
        """
        Print timing statistics for the current step.

        This method calculates and logs the estimated completion time and the remaining
        simulation time based on the current progress and performance metrics.
        """
        step_per_hour = mc_profiler.steps_per_hour
        std_step_per_hour = mc_profiler.std_step_per_hour
        completion_time = mc_profiler.estimate_completion_time(self._i_step + self._i_start, self.steps + 1)
        remaning_time = mc_profiler.estimate_remaining_time(self._i_step + self._i_start, self.steps + 1)
        if remaning_time.days >= 1.0:
            fmt_remaning_time = f"{remaning_time.days} days and {remaning_time.seconds//3600}:{(remaning_time.seconds//60)%60}:{remaning_time.seconds%60}"
        else:
            fmt_remaning_time = (
                f"{remaning_time.seconds//3600}:{(remaning_time.seconds//60)%60}:{remaning_time.seconds%60}"
            )
        logger.info(
            ("-- step/h: " + f"{step_per_hour:.1f} ± {1.96 * std_step_per_hour:<.3f} ".rjust(15, " ")).ljust(40, "-")
            + f" Estimated end: {completion_time.strftime('%Y/%m/%d %H:%M:%S')} (time remaining: {fmt_remaning_time})) ---".rjust(
                80, "-"
            )
        )

    @mc_profiler.track
    def mc_step(self) -> bool:
        """
        Perform a single Monte Carlo step.

        This method selects a move type based on probabilities, executes the corresponding
        move, and returns whether the move was accepted.

        Returns:
            bool: True if the move was accepted, False otherwise.

        Raises:
            InvalidEnsembleAttemptType: If an invalid move type is selected.
        """
        logger.info(f" Step {self._i_step:d} ".center(15, " ").center(120, "-"))
        accepted: bool
        if self._i_step > 0:
            self._move_type = self.rng.choice(
                self.allowed_steps,
                p=self._p,
            )
            if self._move_type == "position":
                accepted = self.engine.attempt_position_change()
            elif self._move_type == "volume":
                accepted = self.engine.attempt_volume_change()
            else:
                logger.error(f"Invalid move type selected: {self._move_type}")
                raise InvalidEnsembleAttemptType(self._move_type)
        else:
            self._move_type = "nothing"
            accepted = self.engine.attempt_nothing()
        self.print_step_time()
        return accepted

    def mc_report(self) -> None:
        """
        Print a detailed simulation performance report.

        This includes statistics such as elapsed time, average step time, and step throughput.
        """
        stats = mc_profiler.get_stats()
        logger.info(" " * 30 + "Elapsed time(hh:mm:ss):".ljust(30, " ") + f"{stats['elapsed_time']}".rjust(30))
        logger.info(
            " " * 30
            + "Step time (s):".ljust(30, " ")
            + f"{stats['mean_step_time']:10.3g} ± {1.96*stats['std_step_time']:.3g}".rjust(30)
        )
        logger.info(" " * 30 + "Step per second (1/s):".ljust(30, " ") + f"{stats['steps_per_second']:10.3f}".rjust(30))
        logger.info(" " * 30 + "Step per hour (1/h):".ljust(30, " ") + f"{stats['steps_per_hour']:10.3f}".rjust(30))
        logger.info(
            " " * 30 + "Step per day (1/d):".ljust(30, " ") + f"{stats['steps_per_hour']*24*1e-6:10.3f}×10^6".rjust(30)
        )
        return
