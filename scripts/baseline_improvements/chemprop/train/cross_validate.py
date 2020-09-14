import csv
import os
from typing import Tuple

import numpy as np

from .run_training import run_training
from scripts.baseline_improvements.chemprop.args import TrainArgs
from scripts.baseline_improvements.chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from scripts.baseline_improvements.chemprop.data import get_task_names
from scripts.baseline_improvements.chemprop.utils import create_logger, makedirs, timeit


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation for a Chemprop model.
    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    #import yaml
   # with open("params.yaml", 'r') as fd:
     #       params = yaml.safe_load(fd)
    #args.target_columns = [params['target_column']]
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_column=args.smiles_column,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns
    )

    # Run training on different random seeds for each fold
    all_scores_rmse = []
    all_scores_r2 = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores_rmse.append(model_scores[0])
        all_scores_r2.append(model_scores[1])
    all_scores_rmse = np.array(all_scores_rmse)
    all_scores_r2 = np.array(all_scores_r2)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores_rmse):
        info(f'\tSeed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')
    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores_r2):
        info(f'\tSeed {init_seed + fold_num} ==> test r2 = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} r2 = {score:.6f}')

    # Report scores across models
    print(all_scores_rmse)
    all_scores_rmse = [[i] for i in all_scores_rmse]
    avg_scores = np.nanmean(all_scores_rmse, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    all_scores_r2 = [[i] for i in all_scores_r2]
    avg_scores = np.nanmean(all_scores_r2, axis=1)  # average score for each model across tasks
    mean_score_r2, std_score_r2 = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test r2 = {mean_score_r2:.6f} +/- {std_score_r2:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'\tOverall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores_rmse[:, task_num]):.6f} +/- {np.nanstd(all_scores_rmse[:, task_num]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', f'Mean {args.metric}', f'Standard deviation {args.metric}'] +
                        [f'Fold {i} {args.metric}' for i in range(args.num_folds)])

        for task_num, task_name in enumerate(args.task_names):
            all_scores_rmse = np.array(all_scores_rmse)
            task_scores = all_scores_rmse[:, task_num]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            writer.writerow([task_name, mean, std] + task_scores.tolist())

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.
    This is the entry point for the command line command :code:`chemprop_train`.
    """
    cross_validate(args=TrainArgs().parse_args())
