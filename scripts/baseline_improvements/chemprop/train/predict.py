from typing import List

import torch
from tqdm import tqdm

from scripts.baseline_improvements.chemprop.args import TrainArgs
from scripts.baseline_improvements.chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from scripts.baseline_improvements.chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        # mol_batch, features_batch = batch.batch_graph(), batch.features()
        no_ring_mol_batch = batch.batch_graph(model_type='no-rings', args=args)
        ring_mol_batch, features_batch = batch.batch_graph(model_type='rings', args=args), batch.features()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(ring_mol_batch, no_ring_mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
