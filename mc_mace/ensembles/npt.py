from pathlib import Path

from loguru import logger
from numpy.random import Generator

from mc_mace.mc import MC
from mc_mace.utils.profiler import MCProfiler

from .abc_ensemble import Ensemble

mc_profiler = MCProfiler()


class NPT(Ensemble):
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

    def start_msg(self) -> None:
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
        logger.info(f" Step {self._i_step:d} ".center(15, " ").center(120, "-"))
        accepted: bool
        if self._i_step > 0:
            self._move_type = self.rng.choice(
                self.allowed_steps,
                p=[self.step_probability[k] for k in self.allowed_steps],
            )
            if self._move_type == "position":
                accepted = self.engine.attempt_position_change()
            elif self._move_type == "volume":
                accepted = self.engine.attempt_volume_change()
        else:
            self._move_type = "nothing"
            accepted = self.engine.attempt_nothing()
        self.print_step_time()
        return accepted

    def mc_report(self) -> None:
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

    def run(self) -> None:
        self.start_msg()
        for self._i_step in range(self._i_start, self.steps + 1):
            accepted = self.mc_step()

            if accepted:
                self.success()
            else:
                self.fail()

            if self.tunning_step is not None:
                self.tuning()
        logger.info(" END ".center(120, "="))
        self.print_report()
