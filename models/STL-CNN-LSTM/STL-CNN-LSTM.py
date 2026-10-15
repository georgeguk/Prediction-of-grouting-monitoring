from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from statsmodels.tsa.seasonal import STL
DT = 5
H = 20
PERIOD = 2
MAX_EPOCH = 150
PATIENCE = 20
LR = 0.001
BATCH = 16
CUTS = [20, 40, 60]
STAGE_PERSISTENCE = 2
EXACT_XU_REPRODUCTION = False
SOURCE_DOI = '10.1016/j.tust.2024.106073'
raw = {}
prefix_cache = {}

def stl(y):
    return STL(np.asarray(y), period=PERIOD, seasonal=7, robust=True, seasonal_deg=0, trend_deg=1).fit().trend.copy()

def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

class TrendModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 12, 3, padding=1), nn.ReLU(), nn.Conv1d(12, 16, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, H))

    def forward(self, x, n):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False))
        return self.head(h[-1])

def samples(ids):
    result = []
    for k in ids:
        target = stl(raw[k])
        cuts = list(range(20, (len(target) - 1) * DT, DT))
        for c in cuts:
            pref = prefix_cache[k, c]
            idx = c // DT
            future = target[idx + 1:idx + H + 1]
            result.append(dict(id=k, cut=c, x=pref, y=future - pref[-1], weight=1 / len(cuts)))
    return result

def collate(rows, mean, std):
    n = torch.tensor([len(r['x']) for r in rows], dtype=torch.long)
    x = torch.zeros(len(rows), int(n.max()) + 2, 1)
    y = torch.zeros(len(rows), H)
    mask = torch.zeros(len(rows), H)
    for i, r in enumerate(rows):
        x[i, :n[i], 0] = torch.tensor((r['x'] - mean) / std, dtype=torch.float32)
        if 'y' in r:
            m = len(r['y'])
            y[i, :m] = torch.tensor(r['y'] / std, dtype=torch.float32)
            mask[i, :m] = 1
    w = torch.tensor([r.get('weight', 1) for r in rows], dtype=torch.float32)
    return (x, n, y, mask, w)

def loss_each(y, t, mask):
    return ((y - t).square() * mask).sum(1) / mask.sum(1).clamp(min=1)

def train_model(train, val, s, fixed_epochs=None):
    supported = [sum((len(r['y']) > h for r in train)) for h in range(H)]
    assert min(supported) > 0, f'Untrained forecast output: {supported}'
    seed(s)
    flat = np.concatenate([r['x'] for r in train])
    mean = float(flat.mean())
    std = max(float(flat.std()), 1e-06)
    model = TrendModel()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    best = np.inf
    best_epoch = 0
    rng = np.random.default_rng(s)
    epochs = fixed_epochs or MAX_EPOCH
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        loss_sum = 0
        for st in range(0, len(order), BATCH):
            batch = [train[j] for j in order[st:st + BATCH]]
            x, n, y, mask, w = collate(batch, mean, std)
            opt.zero_grad()
            le = loss_each(model(x, n), y, mask)
            loss = (le * w).sum() / w.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            loss_sum += float(loss.detach())
        if val:
            model.eval()
            vsum = wsum = 0.0
            with torch.inference_mode():
                for st in range(0, len(val), 64):
                    x, n, y, mask, w = collate(val[st:st + 64], mean, std)
                    vsum += float((loss_each(model(x, n), y, mask) * w).sum())
                    wsum += float(w.sum())
            v = vsum / wsum
            history.append(dict(epoch=epoch, validation_mse=v))
            if v < best - 1e-08:
                best = v
                best_epoch = epoch
            if epoch - best_epoch >= PATIENCE:
                break
        else:
            best_epoch = epoch
    model.eval()
    return (model, mean, std, best_epoch, history)

def forecast(model, mean, std, k, c):
    p = prefix_cache[k, c]
    x, n, _, _, _ = collate([dict(x=p)], mean, std)
    with torch.inference_mode():
        pred = model(x, n)[0].numpy().astype(float) * std + p[-1]
    return pred

def stage(trend, cut, m=2):
    y = np.asarray(trend, float)
    t = np.arange(len(y)) * DT
    if not np.isfinite(y).all():
        return (np.nan, 'nonfinite_forecast')
    bad = np.flatnonzero(y <= 0)
    if len(bad):
        y = y[:bad[0]]
        t = t[:bad[0]]
    g = np.diff(y) / DT
    pos = g > 0
    non = g <= 0
    run = lambda a: [i for i in range(len(a) - m + 1) if np.all(a[i:i + m])]
    for s1 in run(pos):
        for s2 in run(non):
            if s2 < s1 + m:
                continue
            for s3 in run(pos):
                if s3 < s2 + m:
                    continue
                if np.min(y[s2:s3 + 1]) > np.max(y[s1:s2 + 1]):
                    continue
                when = float(t[s3])
                return (when, 'valid') if when >= cut + DT else (np.nan, 'stage3_at_or_before_cutoff')
    return (np.nan, 'nonpositive_before_identification' if len(bad) else 'three_stage_pattern_absent')

def initialize(timeseries):
    global raw, prefix_cache
    torch.set_num_threads(2)
    series = {k: g.sort_values('time_min') for k, g in timeseries.groupby('trajectory_id')}
    raw = {k: g.response_raw.to_numpy(float) / float(g.response_raw.iloc[0]) for k, g in series.items()}
    prefix_cache = {(k, c): stl(y[:c // DT + 1]) for k, y in raw.items() for c in range(20, (len(y) - 1) * DT + 1, DT)}
