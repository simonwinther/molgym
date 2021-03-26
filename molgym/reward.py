import abc
import time
from typing import Tuple, Dict
import math

import ase.data
from ase import Atoms, Atom
from xtb.calculators import GFN2
from ase.calculators.calculator import CalculationFailed
import numpy as np


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self, rho) -> None:
        self.calculator = GFN2()
        self.atom_energies: Dict[str, float] = {}
        self.rho = rho

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start = time.time()

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        try:
            e_tot = self._convert_ev_to_hartree(self._calculate_energy(all_atoms))
            e_parts = self._convert_ev_to_hartree(self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom))
            delta_e = e_tot - e_parts

            elapsed = time.time() - start

            reward = -1 * delta_e

            dist = self._calculate_distance(new_atom)

            reward = reward - (dist * self.rho)
            if math.isnan(reward):
                print('{}'.format(e_tot))
                print('{}'.format(e_parts))
                print('{}'.format(dist))

        except Exception as e:
            reward = -1.00
            elapsed = time.time() - start
        info = {
            'elapsed_time': elapsed,
        }

        if math.isnan(reward):
            reward = -1.00
            info = {'elapsed_time': 'nan' }
        return reward, info

    def _calculate_atomic_energy(self, atom: Atom) -> float:
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            self.atom_energies[atom.symbol] = self._calculate_energy(atoms)
        return self.atom_energies[atom.symbol]

    def _calculate_energy(self, atoms: Atoms) -> float:
        if len(atoms) == 0:
            return 0.0

        # atoms.calc = self.calculator
        atoms.set_calculator(self.calculator)
        return atoms.get_potential_energy()

    def _calculate_distance(self, atom: Atom):
        return np.linalg.norm((0, 0, 0)-atom.position)

    def _convert_ev_to_hartree(self, energy):
        return energy/27.2107
