"""Optimizes hyperparameters using Bayesian optimization."""

from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe
import numpy as np
import sys
sys.path.append('/home/mol2/liza/mol_properties')

print(sys.path)

from scripts.baseline_improvements.chemprop.args import HyperoptArgs
from scripts.baseline_improvements.chemprop.constants import HYPEROPT_LOGGER_NAME
from scripts.baseline_improvements.chemprop.models import MoleculeModel
from scripts.baseline_improvements.chemprop.nn_utils import param_count
from scripts.baseline_improvements.chemprop.train import cross_validate
from scripts.baseline_improvements.chemprop.utils import create_logger, makedirs, timeit


SPACE = {
    'no_substructures_hidden_size': hp.choice('no_substructures_hidden_size', [400, 600]),
    'substructures_hidden_size': hp.choice('substructures_hidden_size', [200, 300]),
    'no_substructures_depth': hp.quniform('no_substructures_depth', low=2, high=7, q=1),
}
INT_KEYS = ['no_substructures_hidden_size', 'substructures_hidden_size', 'no_substructures_depth']


@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args) -> None:
    """
    Runs hyperparameter optimization on a Chemprop model.

    Hyperparameter optimization optimizes the following parameters:

    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    * :code:`ffn_num_layers`: The number of feed-forward layers after message passing is selected from {1, 2, 3}

    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.

    :param args: A :class:`~chemprop.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        hyper_args.ffn_hidden_size = hyper_args.no_substructures_hidden_size + hyper_args.substructures_hidden_size

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        mean_score, std_score = cross_validate(hyper_args)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.RandomState(args.seed))

    # Report best result
    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def chemprop_hyperopt() -> None:
    """Runs hyperparameter optimization for a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_hyperopt`.
    """
    hyperopt(args=HyperoptArgs().parse_args())


if __name__ == "__main__":
    sys.stdout = open('log.txt', 'w')
    try:
        chemprop_hyperopt()
    finally:
        sys.stdout.close()
