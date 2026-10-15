from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
SEED = 20260825
SEEDS = [20260825, 20260826, 20260827, 20260828, 20260829]
MAX_EPOCHS = 120
PATIENCE = 18

def seed_everything(seed: int=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

def trajectory_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(['condition', 'trajectory_id'])['sample_id'].transform('count').to_numpy(float)
    n_trajectories = frame.groupby('condition')['trajectory_id'].transform('nunique').to_numpy(float)
    weights = 1.0 / (counts * n_trajectories)
    return weights / np.mean(weights)

def build_rk_groups(sequence_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = sequence_rows[sequence_rows['time_min'] >= 5].copy()
    valid = valid.rename(columns={'abs_relative_change': 'r_k'})
    return {sample_id: block.sort_values('time_min').reset_index(drop=True) for sample_id, block in valid.groupby('sample_id')}

def rk_features(samples: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for row in samples.itertuples(index=False):
        block = groups[row.sample_id]
        t = block['time_min'].to_numpy(float)
        r = block['r_k'].to_numpy(float)

        def segment(window: float) -> tuple[np.ndarray, np.ndarray]:
            mask = t >= max(5.0, float(row.cutoff_min) - window)
            return (t[mask], r[mask])

        def slope(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
        t10, r10 = segment(10.0)
        t20, r20 = segment(20.0)
        rows.append({'sample_id': row.sample_id, 'cutoff_min': float(row.cutoff_min), 'rk_last': float(r[-1]), 'rk_mean': float(np.mean(r)), 'rk_std': float(np.std(r)), 'rk_min': float(np.min(r)), 'rk_max': float(np.max(r)), 'rk_range': float(np.ptp(r)), 'rk_slope_per_min': slope(t, r), 'rk_recent10_mean': float(np.mean(r10)), 'rk_recent10_std': float(np.std(r10)), 'rk_recent10_slope_per_min': slope(t10, r10), 'rk_recent20_mean': float(np.mean(r20)), 'rk_recent20_std': float(np.std(r20)), 'rk_recent20_slope_per_min': slope(t20, r20), 'rk_cumulative_abs_change': float(np.sum(np.abs(np.diff(r)))) if len(r) > 1 else 0.0})
    derived = pd.DataFrame(rows)
    columns = [column for column in derived.columns if column != 'sample_id']
    return (samples.merge(derived, on='sample_id', validate='one_to_one', suffixes=('', '_derived')), columns)

@dataclass
class SequenceRecord:
    sample_id: str
    values: torch.Tensor
    target: float
    weight: float

class RkDataset(Dataset):

    def __init__(self, records: list[SequenceRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self.records[index]

def collate(records: list[SequenceRecord]):
    lengths = torch.tensor([len(record.values) for record in records], dtype=torch.long)
    values = pad_sequence([record.values for record in records], batch_first=True)
    targets = torch.tensor([record.target for record in records], dtype=torch.float32)
    weights = torch.tensor([record.weight for record in records], dtype=torch.float32)
    return (values, lengths, targets, weights, [record.sample_id for record in records])

class RkCnnBiLstm(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 12, kernel_size=3, padding=1), nn.ReLU(), nn.Conv1d(12, 16, kernel_size=3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(16, 18, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(36, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, 1), nn.Softplus())

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(values.transpose(1, 2)).transpose(1, 2)
        packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat([hidden[-2], hidden[-1]], dim=1)).squeeze(1)

def rk_stats(frame: pd.DataFrame, groups: dict[str, pd.DataFrame]) -> tuple[float, float]:
    values = np.concatenate([groups[sample_id]['r_k'].to_numpy(float) for sample_id in frame['sample_id']])
    mean, std = (float(np.mean(values)), float(np.std(values)))
    return (mean, std if std > 1e-10 else 1.0)

def make_records(frame: pd.DataFrame, groups: dict[str, pd.DataFrame], rk_mean: float, rk_std: float, target_scale: float) -> list[SequenceRecord]:
    weights = trajectory_balanced_weights(frame)
    records = []
    for position, row in enumerate(frame.itertuples(index=False)):
        r_k = groups[row.sample_id]['r_k'].to_numpy(float)
        values = ((r_k - rk_mean) / rk_std)[:, None].astype(np.float32)
        records.append(SequenceRecord(row.sample_id, torch.from_numpy(values), float(row.target_remaining_min) / target_scale, float(weights[position])))
    return records

def weighted_mae(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.abs(prediction - target), weights=weights))

def select_epochs(train: pd.DataFrame, validation: pd.DataFrame, groups: dict[str, pd.DataFrame], seed: int):
    seed_everything(seed)
    rk_mean, rk_std = rk_stats(train, groups)
    target_scale = max(1.0, float(train['target_remaining_min'].median()))
    train_records = make_records(train, groups, rk_mean, rk_std, target_scale)
    validation_records = make_records(validation, groups, rk_mean, rk_std, target_scale)
    train_loader = DataLoader(RkDataset(train_records), batch_size=48, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(RkDataset(validation_records), batch_size=128, shuffle=False, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RkCnnBiLstm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.0002)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    best_epoch, best_mae, stale = (1, math.inf, 0)
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for values, lengths, target, weight, _ in train_loader:
            optimizer.zero_grad()
            prediction = model(values.to(device), lengths)
            loss = (loss_fn(prediction, target.to(device)) * weight.to(device)).sum() / weight.to(device).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        predicted, actual, all_weights = ([], [], [])
        with torch.no_grad():
            for values, lengths, target, weight, _ in validation_loader:
                predicted.append(model(values.to(device), lengths).cpu().numpy() * target_scale)
                actual.append(target.numpy() * target_scale)
                all_weights.append(weight.numpy())
        mae = weighted_mae(np.concatenate(predicted), np.concatenate(actual), np.concatenate(all_weights))
        history.append({'epoch': epoch, 'validation_MAE': mae})
        if mae < best_mae - 0.0001:
            best_epoch, best_mae, stale = (epoch, mae, 0)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return (best_epoch, best_mae, history, sum((parameter.numel() for parameter in model.parameters())))

def fit_cnn_full(train: pd.DataFrame, test: pd.DataFrame, groups: dict[str, pd.DataFrame], epochs: int, seed: int, fold_dir: Path):
    seed_everything(seed)
    rk_mean, rk_std = rk_stats(train, groups)
    target_scale = max(1.0, float(train['target_remaining_min'].median()))
    train_records = make_records(train, groups, rk_mean, rk_std, target_scale)
    test_records = make_records(test, groups, rk_mean, rk_std, target_scale)
    train_loader = DataLoader(RkDataset(train_records), batch_size=48, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(RkDataset(test_records), batch_size=128, shuffle=False, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RkCnnBiLstm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.0002)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    for _ in range(max(1, epochs)):
        model.train()
        for values, lengths, target, weight, _ in train_loader:
            optimizer.zero_grad()
            prediction = model(values.to(device), lengths)
            loss = (loss_fn(prediction, target.to(device)) * weight.to(device)).sum() / weight.to(device).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for values, lengths, _, _, _ in test_loader:
            predictions.append(model(values.to(device), lengths).cpu().numpy() * target_scale)
    prediction = np.maximum(0.0, np.concatenate(predictions))
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'rk_mean': rk_mean, 'rk_std': rk_std, 'target_scale': target_scale, 'epochs': epochs, 'input': 'r_k only; t=5 min onward', 'formula': 'abs((R_NE[k]-R_NE[k-1])/R_NE[k-1])'}, fold_dir / 'cnn_bilstm_rk.pt')
    return (prediction, str(device))

def build_global_ids(samples: pd.DataFrame) -> pd.DataFrame:
    trajectories = samples[['condition', 'trajectory_id', 'point_no']].drop_duplicates().sort_values(['condition', 'point_no'])
    rows, next_id = ([], 1)
    for condition in [1, 2, 3]:
        block = trajectories[trajectories['condition'] == condition]
        for row in block.itertuples(index=False):
            rows.append({'global_point_id': next_id, 'condition': int(condition), 'trajectory_id': row.trajectory_id, 'original_point_no': int(row.point_no)})
            next_id += 1
    mapping = pd.DataFrame(rows)
    expected = {1: (1, 12), 2: (13, 36), 3: (37, 60)}
    for condition, (lower, upper) in expected.items():
        actual = mapping.loc[mapping['condition'] == condition, 'global_point_id']
        if (int(actual.min()), int(actual.max()), len(actual)) != (lower, upper, upper - lower + 1):
            raise RuntimeError(f'global ID mapping failed for condition {condition}')
    return mapping

def make_split(mapping: pd.DataFrame, seed: int, repeat_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    allocation = {1: (8, 2, 2), 2: (16, 4, 4), 3: (16, 4, 4)}
    rows = []
    for condition in [1, 2, 3]:
        ids = mapping.loc[mapping['condition'] == condition, 'global_point_id'].to_numpy()
        rng.shuffle(ids)
        n_train, n_validation, n_test = allocation[condition]
        roles = ['train'] * n_train + ['validation'] * n_validation + ['test'] * n_test
        for global_id, role in zip(ids, roles):
            rows.append({'repeat_id': repeat_id, 'seed': seed, 'global_point_id': int(global_id), 'condition': condition, 'role': role})
    split = pd.DataFrame(rows)
    if split.groupby(['global_point_id'])['role'].nunique().max() != 1:
        raise RuntimeError('group leakage inside split')
    return split

def prediction_frame(test: pd.DataFrame, prediction: np.ndarray, model: str, repeat_id: int) -> pd.DataFrame:
    result = test[['sample_id', 'global_point_id', 'condition', 'trajectory_id', 'point_no', 'cutoff_min', 'b2_target_min', 'target_remaining_min']].copy()
    result['repeat_id'] = repeat_id
    result['model'] = model
    result['pred_remaining'] = np.maximum(0.0, prediction)
    result['pred_b2'] = result['cutoff_min'] + result['pred_remaining']
    result['error_min'] = result['pred_b2'] - result['b2_target_min']
    result['abs_error_min'] = result['error_min'].abs()
    return result
