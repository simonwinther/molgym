import abc
import time
from typing import Tuple, Dict

import ase.data
from ase import Atoms, Atom
from ase.calculators.dftb import Dftb


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self) -> None:
        self.calculator = Dftb(Hamiltonian_='DFTB', Hamiltonian_SCC='Yes',
                               Hamiltonian_SCCTolerance=1e-8,
                               Hamiltonian_MaxAngularMomentum_='',
                               Hamiltonian_MaxAngularMomentum_Br='d',
                               Hamiltonian_MaxAngularMomentum_C='p',
                               Hamiltonian_MaxAngularMomentum_Ca='p',
                               Hamiltonian_MaxAngularMomentum_Cl='d',
                               Hamiltonian_MaxAngularMomentum_F='p',
                               Hamiltonian_MaxAngularMomentum_H='s',
                               Hamiltonian_MaxAngularMomentum_I='d',
                               Hamiltonian_HubbardDerivs_='',
                               Hamiltonian_HubbardDerivs_Br=-0.0573,
                               Hamiltonian_HubbardDerivs_C=-0.1492,
                               Hamiltonian_HubbardDerivs_Ca=-0.0340,
                               Hamiltonian_HubbardDerivs_Cl=-0.0697,
                               Hamiltonian_HubbardDerivs_F=-0.1623,
                               Hamiltonian_HubbardDerivs_H=-0.1857,
                               Hamiltonian_HubbardDerivs_I=-0.0433)

        self.settings = {
            'molecular_charge': 0,
            'max_scf_iterations': 128,
            'unrestricted_calculation': 1,
        }

        self.atom_energies: Dict[str, float] = {}

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start = time.time()

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(
            atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts

        elapsed = time.time() - start

        reward = -1 * delta_e

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

        # self.calculator.set_elements(list(atoms.symbols))
        # self.calculator.set_positions(atoms.positions)
        #self.settings['spin_multiplicity'] = self.get_minimum_spin_multiplicity(atoms)
        # self.calculator.set_settings(self.settings)
        # return self.calculator.calculate_energy()
        atoms.calc = self.calculator
        return atoms.get_potential_energy()
