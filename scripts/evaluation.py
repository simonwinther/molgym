import argparse
import logging

import torch

from molgym.agents.base import AbstractActorCritic
from molgym.agents.internal import InternalAC
from molgym.environment import MolecularEnvironment
from molgym.ppo import ppo
from molgym.reward import InteractionReward
from molgym.spaces import ActionSpace, ObservationSpace
from molgym.tools import mpi, util
from molgym.tools.util import RolloutSaver, InfoSaver, parse_formulas, StructureSaver, load_specific_model
from molgym.buffer import PPOBuffer
from molgym.ppo import rollout

from ase import io

#--loaded_model_name /home/energy/s153999/methanol_generalization/models/CH3OH_6H2O_from_1_run-3.model --clip_ratio 0.25 --target_kl 0.1 --num_steps_per_iter 4096 --bag_refills 5 --formulas H2O --min_mean_distance 0.9 --max_mean_distance 5.6  --seed=1 --num_steps $steps --vf_coef 1 --canvas_size 125

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Agent in MolGym')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=False)
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--model_dir', help='directory for model files', type=str, default='models')
    parser.add_argument('--data_dir', help='directory for saved rollouts', type=str, default='data')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')
    parser.add_argument('--structures_dir', help='directory for traj files of structures', type=str, default='structures')

    # Spaces
    parser.add_argument('--canvas_size',
                        help='maximum number of atoms that can be placed on the canvas',
                        type=int,
                        default=25)
    parser.add_argument('--bag_symbols',
                        help='symbols representing elements in bag (comma separated)',
                        type=str,
                        default='H,He,Li,Be,B,C,N,O,F')

    # Environment
    parser.add_argument('--formulas',
                        help='list of formulas for environment (comma separated)',
                        type=str,
                        required=True)
    parser.add_argument('--eval_formulas',
                        help='list of formulas for environment (comma separated) used for evaluation',
                        type=str,
                        required=False)
    parser.add_argument('--min_atomic_distance', help='minimum allowed atomic distance', type=float, default=0.6)
    parser.add_argument('--max_h_distance',
                        help='maximum distance a H atom can be away from the nearest heavy atom',
                        type=float,
                        default=2.0)
    parser.add_argument('--min_reward', help='minimum reward given by environment', type=float, default=-0.6)

    parser.add_argument('--rho', help='', type=float, default=0.01)
    parser.add_argument('--bag_refills', help='', type=int, default=5)
    parser.add_argument('--starting_canvas', help='', type=str, default='')

    # Model
    parser.add_argument('--min_mean_distance', help='minimum mean distance', type=float, default=0.8)
    parser.add_argument('--max_mean_distance', help='maximum mean distance', type=float, default=1.8)
    parser.add_argument('--network_width', help='width of FC layers', type=int, default=128)

    parser.add_argument('--load_model', help='load latest checkpoint file', action='store_true', default=False)
    parser.add_argument('--loaded_model_name', help='load specific model ', default=None)
    parser.add_argument('--save_freq', help='save model every <n> iterations', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluate model every <n> iterations', type=int, default=5)
    parser.add_argument('--num_eval_episodes', help='number of episodes per evaluation', type=int, default=None)

    # Training algorithm
    parser.add_argument('--discount', help='discount factor', type=float, default=1.0)
    parser.add_argument('--num_steps', dest='max_num_steps', help='maximum number of steps', type=int, default=50000)
    parser.add_argument('--num_steps_per_iter',
                        help='number of optimization steps per iteration',
                        type=int,
                        default=128)
    parser.add_argument('--clip_ratio', help='PPO clip ratio', type=float, default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate of Adam optimizer', type=float, default=3e-4)
    parser.add_argument('--vf_coef', help='Coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--entropy_coef', help='Coefficient for entropy loss', type=float, default=0.01)
    parser.add_argument('--max_num_train_iters', help='Maximum number of training iterations', type=int, default=7)
    parser.add_argument('--gradient_clip', help='maximum norm of gradients', type=float, default=0.5)
    parser.add_argument('--lam', help='Lambda for GAE-Lambda', type=float, default=0.97)
    parser.add_argument('--target_kl',
                        help='KL divergence between new and old policies after an update for early stopping',
                        type=float,
                        default=0.01)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')
    parser.add_argument('--save_rollouts',
                        help='which rollouts to save',
                        type=str,
                        default='none',
                        choices=['none', 'train', 'eval', 'all'])
    parser.add_argument('--all_ranks', help='print log of all ranks', action='store_true', default=False)

    return parser.parse_args()


def get_config() -> dict:
    config = vars(parse_args())

    config['num_procs'] = mpi.get_num_procs()

    return config


def main() -> None:
    config = get_config()

    bag_symbols = config['bag_symbols'].split(',')
    action_space = ActionSpace()
    observation_space = ObservationSpace(canvas_size=config['canvas_size'], symbols=bag_symbols)

    model = load_specific_model(model_path=config['loaded_model_name'])
    model.action_space = action_space
    model.observation_space = observation_space

    reward = InteractionReward(config['rho'])

    if not config['eval_formulas']:
        config['eval_formulas'] = config['formulas']

    eval_formulas = parse_formulas(config['eval_formulas'])

    eval_init_formulas = parse_formulas(config['eval_formulas'])

    eval_env = MolecularEnvironment(
        reward=reward,
        observation_space=observation_space,
        action_space=action_space,
        formulas=eval_formulas,
        min_atomic_distance=config['min_atomic_distance'],
        max_h_distance=config['max_h_distance'],
        min_reward=config['min_reward'],
        initial_formula=eval_init_formulas,
        bag_refills=config['bag_refills'],
    )

    eval_buffer_size = 1000
    eval_buffer = PPOBuffer(int_act_dim=model.internal_action_dim, size=eval_buffer_size, gamma=config['discount'], lam=config['lam'])

    with torch.no_grad():
        model.training = False
        rollout_info = rollout(model, eval_env, eval_buffer, num_episodes=1)
        model.training = True
        logging.info('Evaluation rollout: ' + str(rollout_info))

        atoms, _ = eval_env.observation_space.parse(eval_buffer.next_obs_buf[eval_buffer.ptr-1])
        print(atoms)
        io.write('/home/energy/s153999/evaluated_structure.traj', atoms)

if __name__ == '__main__':
    main()
