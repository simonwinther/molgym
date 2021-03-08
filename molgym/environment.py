import abc
import itertools
from typing import Tuple, List

import ase.formula
from ase.build import molecule
import gym
import numpy as np
from ase import Atoms, Atom

from molgym.reward import InteractionReward
from molgym.spaces import ActionSpace, ObservationSpace, ActionType, ObservationType, NULL_SYMBOL
from molgym.tools import util


class AbstractMolecularEnvironment(gym.Env, abc.ABC):
    def __init__(
        self,
        reward: InteractionReward,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        initial_formula,
        min_atomic_distance=0.6,  # Angstrom
        max_h_distance=2.0,  # Angstrom
        min_reward=-0.6,  # Hartree
        bag_refills=5,
    ):
        self.reward = reward
        self.observation_space = observation_space
        self.action_space = action_space

        self.random_state = np.random.RandomState()

        self.min_atomic_distance = min_atomic_distance
        self.max_h_distance = max_h_distance
        self.min_reward = min_reward

        self.current_atoms = Atoms()
        self.current_formula = ase.formula.Formula()

        self.bag_refills = bag_refills
        self.initial_formula = initial_formula
        print(self.initial_formula[-1].count())
    # Negative reward should be on the same order of magnitude as the positive ones.
    # Memory agent on QM9: mean 0.26, std 0.13, min -0.54, max 1.23 (negative reward indeed possible
    # but avoidable and probably due to PM6)

    @abc.abstractmethod
    def reset(self) -> ObservationType:
        raise NotImplementedError

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        new_atom = self.action_space.to_atom(action)
        done = new_atom.symbol == NULL_SYMBOL

        if done:
            return self.observation_space.build(self.current_atoms, self.current_formula), 0.0, done, {}

        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                self.min_reward,
                True,
                {},
            )

        reward, info = self.reward.calculate(self.current_atoms, new_atom)

        if reward < self.min_reward:
            done = True
            reward = self.min_reward

        self.current_atoms.append(new_atom)
        self.current_formula = util.remove_from_formula(self.current_formula, new_atom.symbol)

        if len(self.current_formula) == 0 and self.bag_refills > 0:
            self.current_formula = self.initial_formula[-1]
            self.bag_refills -= 1

        # Check if state is terminal
        if self._is_terminal():
            done = True

        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def _is_terminal(self) -> bool:
        return len(self.current_atoms) == self.observation_space.canvas_space.size or len(self.current_formula) == 0

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False

        return self._all_h_covered(current_atoms, new_atom)

    def _is_too_close(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Check distances between new and old atoms
        for existing_atom in existing_atoms:
            if np.linalg.norm(existing_atom.position - new_atom.position) < self.min_atomic_distance:
                return True

        return False

    def _all_h_covered(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Ensure that H atoms are not too far away from the nearest heavy atom
        if len(existing_atoms) == 0 or new_atom.symbol != 'H':
            return True

        for existing_atom in existing_atoms:
            if existing_atom.symbol == 'H':
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            if distance < self.max_h_distance:
                return True

        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None) -> int:
        seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(seed)
        return seed


class MolecularEnvironment(AbstractMolecularEnvironment):
    def __init__(self, formulas: List[ase.formula.Formula], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formulas_cycle = itertools.cycle(formulas)
        self.reset()

    def reset(self) -> ObservationType:
        self.current_atoms = molecule('F')
        self.current_formula = next(self.formulas_cycle)
        self.bag_refills = 5
        return self.observation_space.build(self.current_atoms, self.current_formula)
