import numpy as np
from ase import Atoms
from ase.neighborlist import NewPrimitiveNeighborList, first_neighbors
from vesin import ase_neighbor_list


class FastPrimitiveNeighborList(NewPrimitiveNeighborList):  # type: ignore[misc]
    """
    A fast implementation of a primitive neighbor list, wrapping around
    ASE's `NewPrimitiveNeighborList` and additional utilities.

    This class determines neighbors for atoms based on specified cutoff radii
    and provides optimized methods for updating and building the neighbor list.

    Args:
        cutoffs (list[float]): List of cutoff radii, one for each atom. If the
            spheres (defined by their cutoff radii) of two atoms overlap, they
            will be counted as neighbors.
        skin (float, optional): Skin distance for reusing the neighbor list
            without rebuilding. Default is 0.3.
        sorted (bool, optional): If True, the neighbor list is sorted. Default
            is False.
        self_interaction (bool, optional): If True, atoms can list themselves
            as neighbors. Default is True.
        bothways (bool, optional): If True, neighbors are considered in both
            directions. Default is False.
        use_scaled_positions (bool, optional): Whether to use scaled positions
            for calculations. Default is False.
    """

    def __init__(
        self,
        cutoffs: list[float],
        skin: float = 0.3,
        sorted: bool = False,
        self_interaction: bool = True,
        bothways: bool = False,
        use_scaled_positions: bool = False,
    ) -> None:
        super().__init__(
            cutoffs,
            skin=skin,
            sorted=sorted,
            self_interaction=self_interaction,
            bothways=bothways,
            use_scaled_positions=use_scaled_positions,
        )

    def update(
        self,
        pbc: np.ndarray,
        cell: np.ndarray,
        positions: np.ndarray,
        numbers: np.ndarray = None,
    ) -> bool:
        """
        Update the neighbor list if necessary.

        Args:
            pbc (np.ndarray): Periodic boundary conditions.
            cell (np.ndarray): Simulation cell matrix.
            positions (np.ndarray): Atom positions.
            numbers (np.ndarray, optional): Atomic numbers. Default is None.

        Returns:
            bool: True if the list was rebuilt, False otherwise.
        """
        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        if (
            (self.pbc != pbc).any()
            or (self.cell != cell).any()
            or ((self.positions - positions) ** 2).sum(1).max() > self.skin**2
        ):
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        return False

    def build(
        self,
        pbc: np.ndarray,
        cell: np.ndarray,
        positions: np.ndarray,
        numbers: np.ndarray = None,
    ) -> None:
        """
        Build the neighbor list from the given atomic data.

        Args:
            pbc (np.ndarray): Periodic boundary conditions.
            cell (np.ndarray): Simulation cell matrix.
            positions (np.ndarray): Atom positions.
            numbers (np.ndarray, optional): Atomic numbers. Default is None.
        """
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)

        atoms_ = Atoms(["X"] * positions.shape[0], positions=positions, cell=cell, pbc=pbc)

        pair_first, pair_second, offset_vec = ase_neighbor_list(
            "ijS", atoms_, self.cutoffs, self_interaction=self.self_interaction
        )

        if len(positions) > 0 and not self.bothways:
            offset_x, offset_y, offset_z = offset_vec.T

            mask = offset_z > 0
            mask &= offset_y == 0
            mask |= offset_y > 0
            mask &= offset_x == 0
            mask |= offset_x > 0
            mask |= (pair_first <= pair_second) & (offset_vec == 0).all(axis=1)

            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(pair_first * len(pair_first) + pair_second)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        self.pair_first = pair_first
        self.pair_second = pair_second
        self.offset_vec = offset_vec

        # Compute the index array pointing to the first neighbor
        self.first_neigh = first_neighbors(len(positions), pair_first)

        self.nupdates += 1
