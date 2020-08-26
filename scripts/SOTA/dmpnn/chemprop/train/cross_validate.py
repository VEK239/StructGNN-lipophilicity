import csv
import os
from typing import Tuple

import numpy as np

from .run_training import run_training
from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_task_names
from chemprop.utils import create_logger, makedirs, timeit

import json


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
    import yaml

    with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)
    args.save_dir = params['save_dir']
    args.epochs = params['epochs']
    args.depth = params['depth']
    args.features_generator = [params['features_generator']]
    args.no_features_scaling = params['no_features_scaling']
    args.split_type = params['split_type']
    args.num_folds = params['num_folds']
    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_column=args.smiles_column,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns
    )
    # Run training on different random seeds for each fold
    all_scores = []
    all_scores_r2 = []
    val_scores = []
    val_scores_r2 = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores, model_scores_r2, val_score, val_score_r2 = run_training(args, logger)
        all_scores.append(model_scores)
        all_scores_r2.append(model_scores_r2)
        val_scores.append(val_score)
        val_scores_r2.append(val_score_r2)
    all_scores = np.array(all_scores)
    all_scores_r2 = np.array(all_scores_r2)
    val_scores = np.array(val_scores)
    val_scores_r2 = np.array(val_scores_r2)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'\tSeed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')
        info(f'\tSeed {init_seed + fold_num} ==> test R2 = {np.nanmean(all_scores_r2[fold_num]):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(args.task_names, scores):
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')
                info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} R2 = {all_scores_r2[fold_num]:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    
#     avg_val_scores = np.nanmean(val_scores)  # average score for each model across tasks
    mean_val_score, std_val_score = np.nanmean(val_scores), np.nanstd(val_scores)
    
    avg_scores_r2 = np.nanmean(all_scores_r2, axis=1)  # average score for each model across tasks
    mean_score_r2, std_score_r2 = np.nanmean(avg_scores_r2), np.nanstd(avg_scores_r2)
    
#     avg_val_scores_r2 = np.nanmean(val_scores_r2, axis=1)  # average score for each model across tasks
    mean_val_score_r2, std_val_score_r2 = np.nanmean(val_scores_r2), np.nanstd(val_scores_r2)
    
    info(f'Overall val {args.metric} = {mean_val_score:.6f} +/- {std_val_score:.6f}')
    info(f'Overall val R2 = {mean_val_score_r2:.6f} +/- {std_val_score_r2:.6f}')
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
    info(f'Overall test R2 = {mean_score_r2:.6f} +/- {std_score_r2:.6f}')
    
    all_scores_dict = {}
    all_scores_dict['test_RMSE_mean'] = mean_score
    all_scores_dict['test_R2_mean'] = mean_score_r2
    all_scores_dict['test_RMSE_std'] = std_score
    all_scores_dict['test_R2_std'] = std_score_r2
    all_scores_dict['val_RMSE_mean'] = mean_val_score
    all_scores_dict['val_R2_mean'] = mean_val_score_r2
    all_scores_dict['val_RMSE_std'] = std_val_score
    all_scores_dict['val_R2_std'] = std_val_score_r2
    
    with open(os.path.join(os.path.dirname(save_dir), 'final_scores.json'), 'w') as f:
        json.dump(all_scores_dict, f)
    
    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'\tOverall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    # Save scores
    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', f'Mean {args.metric}', f'Standard deviation {args.metric}'] +
                        [f'Fold {i} {args.metric}' for i in range(args.num_folds)])

        for task_num, task_name in enumerate(args.task_names):
            task_scores = all_scores[:, task_num]
            mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
            writer.writerow([task_name, mean, std] + task_scores.tolist())

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    cross_validate(args=TrainArgs().parse_args())
