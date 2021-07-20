import abc
import time
from typing import Tuple, Dict
import math

import ase.data
from ase import Atoms, Atom
from xtb.calculators import GFN2
from ase.calculators.calculator import CalculationFailed
import numpy as np
import ase.io

class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self, rho) -> None:
        self.calculator = GFN2(max_iterations=50)
        self.atom_energies: Dict[str, float] = {}
        self.rho = rho

    def calculate(self, atoms: Atoms, new_atom: Atom, min_reward: float, bag_length: int, refills: int) -> Tuple[float, dict]:
        start = time.time()

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        #if bag_length == 0 and refills == 0: # final reward
        try:
            e_tot = self._calculate_energy(all_atoms)
            e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
            delta_e = self._convert_ev_to_hartree(e_tot) - self._convert_ev_to_hartree(e_parts)

            elapsed = time.time() - start

            reward = -1 * delta_e

            dist = self._calculate_distance(new_atom)

            reward = reward - (dist * self.rho)

            # Used for final reward
            #reward = (-1 * self._convert_ev_to_hartree(e_tot) / len(all_atoms)) - (dist * self.rho)

            if math.isnan(reward):
                reward = min_reward

            if reward > 20:
                reward = -1.00

        except Exception as e:
            reward = min_reward
            elapsed = time.time() - start
        
        #else: # final reward
        #    reward = 0 # final reward
        #    elapsed = time.time() - start

        if math.isnan(reward):
            reward = min_reward
        
        info = {
                'elapsed_time': elapsed,
            }
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
