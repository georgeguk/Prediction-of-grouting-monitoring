from pathlib import Path
import argparse, sys, shutil, importlib.util
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parent

def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

cnn = load_module('cnn_bilstm', 'CNN-BiLSTM/CNN-BiLSTM.py')
xu = load_module('stl_cnn_lstm', 'STL-CNN-LSTM/STL-CNN-LSTM.py')
gmm = load_module('gmm', 'GMM.py')
LAB_INTERVAL_MIN = 5
FIELD_INTERVAL_MIN = 20
START_MIN, WIN, MIN_LEN = (10.0, 5, 3)
CUTS = [20, 40, 60]
FIELD_INPUT_ROUNDS = [4, 5, 6, 7]
SITE_TERMINATION_MIN = {'A1': 163.0, 'A2': 158.0}
DEPENDENCIES = {'numpy': '2.2.2', 'pandas': '2.2.3', 'scipy': '1.15.3', 'torch': '2.9.1', 'scikit-learn': '1.7.1', 'statsmodels': '0.14.6', 'openpyxl': '3.1.5', 'threadpoolctl': '3.6.0'}

def read_energy(book):
    sheets = pd.read_excel(book, sheet_name=None)
    if len(sheets) != 4:
        raise ValueError('Expected three laboratory sheets and one field sheet')
    laboratory, field = ([], [])
    for condition, frame in enumerate(list(sheets.values())[:3], 1):
        cols = [c for c in frame.columns if c == 'Baseline' or str(c).startswith('Round ')]
        for _, row in frame.iterrows():
            values = row[cols].to_numpy(float)
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Laboratory energy ratios must be positive and complete')
            point = int(str(row['Point']).split('-')[-1])
            laboratory.append(dict(condition=condition, point_no=point, trajectory_id=f'C{condition}_P{point:02d}', region=row['Region'], energy=values))
    frame = list(sheets.values())[3]
    cols = [c for c in frame.columns if c == 'Baseline' or str(c).startswith('Round ')]
    for _, row in frame.iterrows():
        values = row[cols].to_numpy(float)
        n = np.flatnonzero(np.isfinite(values))
        if len(n) == 0 or not np.array_equal(n, np.arange(len(n))):
            raise ValueError('Only trailing missing field rounds are allowed')
        values = values[:len(n)]
        if np.any(values <= 0):
            raise ValueError('Field energy ratios must be positive')
        field.append(dict(hole=row['Hole'], point=row['Point'], source_id=int(row['Source ID']), radius_m=float(row['Radius (m)']), included=bool(row['Included']), energy=values))
    return (laboratory, field)

def prepare(laboratory):
    series, labels, samples, sequences = ([], [], [], [])
    for record in sorted(laboratory, key=lambda r: (r['condition'], r['point_no'])):
        if record['region'] != 'Void':
            continue
        condition, point, ident = (record['condition'], record['point_no'], record['trajectory_id'])
        energy = record['energy']
        times = np.arange(len(energy), dtype=float) * LAB_INTERVAL_MIN
        rates = np.r_[np.nan, np.abs(np.diff(energy) / energy[:-1])]
        label, _ = analyze(condition, ident, times[1:], rates[1:])
        b2 = label['stab_start_b2_min']
        labels.append(dict(condition=condition, trajectory_id=ident, point_no=point, b1_min=label['fluc_start_min'], b2_target_min=b2, observation_end_min=times[-1]))
        for t, energy_value, rate in zip(times, energy, rates):
            series.append(dict(condition=condition, trajectory_id=ident, point_no=point, time_min=t, response_raw=energy_value, abs_relative_change=rate, b2_target_min=b2))
        cuts = list(range(20, int(b2), LAB_INTERVAL_MIN))
        for cut in cuts:
            ident_sample = f'{ident}_T{cut:03d}'
            samples.append(dict(sample_id=ident_sample, condition=condition, trajectory_id=ident, point_no=point, cutoff_min=cut, b2_target_min=b2, target_remaining_min=b2 - cut, n_steps=cut // LAB_INTERVAL_MIN + 1, trajectory_weight=1 / len(cuts), observation_end_min=times[-1]))
            for index in range(cut // LAB_INTERVAL_MIN + 1):
                sequences.append(dict(sample_id=ident_sample, condition=condition, trajectory_id=ident, cutoff_min=cut, time_min=times[index], abs_relative_change=0.0 if index == 0 else rates[index]))
    samples, sequences = (pd.DataFrame(samples), pd.DataFrame(sequences))
    mapping = cnn.build_global_ids(samples)
    samples = samples.merge(mapping[['trajectory_id', 'global_point_id']], on='trajectory_id', validate='many_to_one')
    splits = pd.concat([cnn.make_split(mapping, seed, rep) for rep, seed in enumerate(cnn.SEEDS, 1)], ignore_index=True)
    return (pd.DataFrame(series), pd.DataFrame(labels), samples, sequences, mapping, splits)

def predict_cnn(samples, sequences, splits, model_root):
    cnn.seed_everything()
    groups = cnn.build_rk_groups(sequences)
    rows = []
    for rep in range(1, 6):
        ids = splits[(splits.repeat_id == rep) & (splits.role == 'test')].global_point_id
        test = samples[samples.global_point_id.isin(ids)].copy()
        ck = torch.load(model_root / 'CNN-BiLSTM' / 'PT' / f'repeat_{rep}.pt', map_location='cpu', weights_only=True)
        model = cnn.RkCnnBiLstm()
        model.load_state_dict(ck['state_dict'])
        model.eval()
        records = cnn.make_records(test, groups, ck['rk_mean'], ck['rk_std'], ck['target_scale'])
        predictions = []
        with torch.inference_mode():
            for values, lengths, *_ in DataLoader(cnn.RkDataset(records), batch_size=128, shuffle=False, collate_fn=cnn.collate):
                predictions.extend((model(values, lengths).numpy() * ck['target_scale']).tolist())
        rows.append(cnn.prediction_frame(test, np.array(predictions), 'CNN-BiLSTM', rep))
    return pd.concat(rows, ignore_index=True)

def train_cnn(samples, sequences, splits, output):
    groups = cnn.build_rk_groups(sequences)
    rows, histories = ([], [])
    for rep, seed in enumerate(cnn.SEEDS, 1):
        role = samples.global_point_id.map(splits[splits.repeat_id == rep].set_index('global_point_id').role)
        train, val, test = [samples[role == r].copy() for r in ['train', 'validation', 'test']]
        epochs, score, history, nparams = cnn.select_epochs(train, val, groups, seed + 1000)
        dest = output / 'models' / 'CNN-BiLSTM' / 'PT'
        predictions, device = cnn.fit_cnn_full(pd.concat([train, val]), test, groups, epochs, seed + 1000, dest / f'repeat_{rep}')
        shutil.copyfile(dest / f'repeat_{rep}' / 'cnn_bilstm_rk.pt', dest / f'repeat_{rep}.pt')
        rows.append(cnn.prediction_frame(test, predictions, 'CNN-BiLSTM', rep))
        histories.extend([dict(repeat_id=rep, **row) for row in history])
        print(f'CNN-BiLSTM repeat {rep}: epoch={epochs}, validation MAE={score:.6f}, parameters={nparams}, device={device}', flush=True)
    pd.DataFrame(histories).to_csv(output / 'cnn_validation_history.csv', index=False)
    return pd.concat(rows, ignore_index=True)

def predict_xu(series, labels, mapping, splits, model_root, output=None, retrain=False):
    xu.initialize(series)
    parts = splits.merge(mapping[['global_point_id', 'trajectory_id']], on='global_point_id', validate='many_to_one')
    rows = []
    label_map = labels.set_index('trajectory_id')
    for rep in range(1, 6):
        p = parts[parts.repeat_id == rep]
        train, val, test = [p[p.role == role].trajectory_id.tolist() for role in ['train', 'validation', 'test']]
        if retrain:
            _, _, _, epochs, history = xu.train_model(xu.samples(train), xu.samples(val), 20260909 + rep)
            model, mean, std, _, _ = xu.train_model(xu.samples(train + val), [], 20261909 + rep, epochs)
            dest = output / 'models' / 'STL-CNN-LSTM' / 'PT'
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std, period=xu.PERIOD, horizon_steps=xu.H, selected_epochs=epochs, fit_ids=train + val, test_ids=test), dest / f'repeat_{rep}.pt')
            pd.DataFrame(history).to_csv(output / f'xu_validation_repeat_{rep}.csv', index=False)
            print(f'STL-CNN-LSTM repeat {rep}: epoch={epochs}', flush=True)
        else:
            ck = torch.load(model_root / 'STL-CNN-LSTM' / 'PT' / f'repeat_{rep}.pt', map_location='cpu', weights_only=True)
            if set(ck['test_ids']) != set(test) or set(ck['fit_ids']) != set(train + val):
                raise ValueError('Checkpoint partition does not match regenerated partition')
            model = xu.TrendModel()
            model.load_state_dict(ck['state_dict'])
            model.eval()
            mean, std = (ck['mean'], ck['std'])
        for ident in test:
            ref = label_map.loc[ident]
            for cut in CUTS:
                if ref.b2_target_min < cut + LAB_INTERVAL_MIN:
                    continue
                fc = xu.forecast(model, mean, std, ident, cut)
                pred, status = xu.stage(np.r_[xu.prefix_cache[ident, cut], fc], cut, xu.STAGE_PERSISTENCE)
                rows.append(dict(repeat_id=rep, trajectory_id=ident, condition=ref.condition, point_no=ref.point_no, cutoff_min=cut, b2_target_min=ref.b2_target_min, model='STL-CNN-LSTM', pred_b2=pred, status=status, valid=status == 'valid'))
    return pd.DataFrame(rows)

def metric_row(frame, paired=False):
    good = np.isfinite(frame.pred_b2.to_numpy(float))
    error = frame.pred_b2.to_numpy(float) - frame.b2_target_min.to_numpy(float)
    if paired:
        frame = frame[good]
        error = error[good]
        good = np.ones(len(frame), dtype=bool)
    if len(frame) == 0:
        return dict(n_eligible=0, n_valid=0, MAE=np.nan, RMSE=np.nan, coverage=np.nan, within5_conditional=np.nan, within10_conditional=np.nan, within5_overall=np.nan)
    weights = 1 / frame.groupby('trajectory_id').trajectory_id.transform('count').to_numpy(float)
    return dict(n_eligible=len(frame), n_valid=int(good.sum()), MAE=float(np.average(abs(error[good]), weights=weights[good])) if good.any() else np.nan, RMSE=float(np.sqrt(np.average(error[good] ** 2, weights=weights[good]))) if good.any() else np.nan, coverage=float(np.average(good, weights=weights)), within5_conditional=float(np.average(abs(error[good]) <= 5, weights=weights[good])) if good.any() else np.nan, within10_conditional=float(np.average(abs(error[good]) <= 10, weights=weights[good])) if good.any() else np.nan, within5_overall=float(np.sum(weights[good] * (abs(error[good]) <= 5)) / weights.sum()))

def evaluate(cnn_predictions, xu_predictions, output):
    rows = []
    for rep, frame in cnn_predictions.groupby('repeat_id'):
        rows.append(dict(scope='all_prefixes_SSE', repeat_id=rep, model='CNN-BiLSTM', cutoff_min=0, **metric_row(frame)))
    a = cnn_predictions[cnn_predictions.cutoff_min.isin(CUTS)]
    all_predictions = pd.concat([a, xu_predictions], ignore_index=True)
    for (rep, cut, method), frame in all_predictions.groupby(['repeat_id', 'cutoff_min', 'model']):
        rows.append(dict(scope='eligible_SSE', repeat_id=rep, model=method, cutoff_min=cut, **metric_row(frame)))
    keys = ['repeat_id', 'trajectory_id', 'cutoff_min']
    valid_keys = xu_predictions[np.isfinite(xu_predictions.pred_b2)][keys]
    paired = all_predictions.merge(valid_keys, on=keys, validate='many_to_one')
    for (rep, cut, method), frame in paired.groupby(['repeat_id', 'cutoff_min', 'model']):
        rows.append(dict(scope='paired_SSE', repeat_id=rep, model=method, cutoff_min=cut, **metric_row(frame)))
    per = pd.DataFrame(rows)
    per.to_csv(output / 'metrics_by_repeat.csv', index=False)
    metrics = ['MAE', 'RMSE', 'coverage', 'within5_conditional', 'within10_conditional', 'within5_overall']
    summary = per.groupby(['scope', 'cutoff_min', 'model'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / 'metrics_summary.csv', index=False)
    print(summary[['MAE_mean', 'RMSE_mean']].to_string())

def predict_field(field, model_root, output):
    torch.set_num_threads(2)
    rows = []
    for rep in range(1, 6):
        ck = torch.load(model_root / 'CNN-BiLSTM' / 'PT' / f'repeat_{rep}.pt', map_location='cpu', weights_only=True)
        model = cnn.RkCnnBiLstm()
        model.load_state_dict(ck['state_dict'])
        model.eval()
        for hole in ['A1', 'A2']:
            points = [p for p in field if p['hole'] == hole]
            for round_no in FIELD_INPUT_ROUNDS:
                raw = np.array([np.abs(np.diff(p['energy'][:round_no + 1]) / p['energy'][:round_no]) for p in points])
                x = (torch.tensor(raw, dtype=torch.float32).unsqueeze(-1) - float(ck['rk_mean'])) / float(ck['rk_std'])
                x = torch.cat([x, torch.zeros(len(x), 2, 1)], dim=1)
                with torch.inference_mode():
                    remaining = model(x, torch.full((len(x),), round_no, dtype=torch.long)).numpy().astype(float) * float(ck['target_scale'])
                for p, rem in zip(points, remaining):
                    cut = round_no * FIELD_INTERVAL_MIN
                    rows.append(dict(hole=hole, point=p['point'], source_id=p['source_id'], radius_m=p['radius_m'], included=p['included'], repeat_id=rep, input_round=round_no, cutoff_min=cut, pred_b2=cut + rem, protocol='Direct20'))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'field_predictions.csv', index=False)
    keys = ['hole', 'point', 'source_id', 'radius_m', 'included', 'input_round', 'cutoff_min']
    summary = frame.groupby(keys, as_index=False).pred_b2.agg(['mean', 'std', 'min', 'max']).reset_index()
    summary.to_csv(output / 'field_predictions_summary.csv', index=False)
    evaluation = []
    for (hole, round_no, rep), g in frame[frame.included & np.isclose(frame.radius_m, 4.5)].groupby(['hole', 'input_round', 'repeat_id']):
        e = g.pred_b2.to_numpy() - SITE_TERMINATION_MIN[hole]
        evaluation.append(dict(hole=hole, input_round=round_no, repeat_id=rep, reference='site_termination_not_point_SSE', reference_min=SITE_TERMINATION_MIN[hole], n_points=len(g), MAE=float(abs(e).mean()), RMSE=float(np.sqrt((e * e).mean()))))
    pd.DataFrame(evaluation).to_csv(output / 'field_outer_ring_metrics.csv', index=False)
    ensemble_rows = []
    outer = summary[summary.included & np.isclose(summary.radius_m, 4.5)].copy()
    outer['reference_min'] = outer.hole.map(SITE_TERMINATION_MIN)
    outer['error_min'] = outer['mean'] - outer.reference_min
    for round_no, group in outer.groupby('input_round'):
        for hole, selected in list(group.groupby('hole')) + [('Overall', group)]:
            errors = selected.error_min.to_numpy(float)
            ensemble_rows.append(dict(hole=hole, input_round=round_no, cutoff_min=round_no * FIELD_INTERVAL_MIN, n_points=len(selected), prediction_aggregation='mean_of_five_models_before_error', reference='site_termination_not_point_SSE', MAE=float(np.abs(errors).mean()), RMSE=float(np.sqrt(np.mean(errors ** 2)))))
    pd.DataFrame(ensemble_rows).to_csv(output / 'field_ensemble_outer_ring_metrics.csv', index=False)

def xu_input_series(series, training_data):
    archived = pd.read_excel(training_data, sheet_name=0, usecols=['trajectory_id', 'time_min', 'response_raw'])
    current = series[['trajectory_id', 'time_min', 'response_raw']]
    joined = current.merge(archived, on=['trajectory_id', 'time_min'], suffixes=('_workbook', '_snapshot'), how='outer', indicator=True, validate='one_to_one')
    if not (joined['_merge'] == 'both').all() or not np.allclose(joined.response_raw_workbook, joined.response_raw_snapshot, rtol=0, atol=1e-14):
        raise ValueError('STL training data does not match workbook records. Use --xu-input workbook for new data.')
    print('STL input: STL training data.xlsx.', flush=True)
    return archived

def main():
    parser = argparse.ArgumentParser(epilog='Dependencies: ' + ', '.join((f'{k}=={v}' for k, v in DEPENDENCIES.items())))
    parser.add_argument('command', nargs='?', default='all', choices=['all', 'prepare', 'gmm', 'cnn', 'xu', 'field', 'train-cnn', 'train-xu'])
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'Original data.xlsx')
    parser.add_argument('--models', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'results')
    parser.add_argument('--xu-input', choices=['training', 'snapshot', 'workbook'], default='training', help='Use STL training data.xlsx, or Original data.xlsx for new experiments. The snapshot option is a legacy alias for training.')
    parser.add_argument('--xu-training-data', '--xu-snapshot', dest='xu_training_data', type=Path, default=ROOT / 'STL-CNN-LSTM' / 'STL training data.xlsx')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    laboratory, field = read_energy(args.data)
    series, labels, samples, sequences, mapping, splits = prepare(laboratory)
    labels.to_csv(args.output / 'sse_labels.csv', index=False)
    mapping.to_csv(args.output / 'point_mapping.csv', index=False)
    splits.to_csv(args.output / 'split_assignments.csv', index=False)
    if args.command in ['all', 'gmm']:
        energy = np.array([r['energy'] for r in laboratory if r['condition'] == 1])
        model, threshold, rates = gmm.fit(energy)
        pd.DataFrame([dict(seed=gmm.SEED, n_points=len(energy), n_values=len(energy) * 7, threshold_r=threshold, threshold_percent=threshold * 100, converged=model.converged_)]).to_csv(args.output / 'gmm_threshold.csv', index=False)
        print('GMM threshold:', threshold)
    if args.command in ['all', 'cnn', 'train-cnn']:
        cp = train_cnn(samples, sequences, splits, args.output) if args.command == 'train-cnn' else predict_cnn(samples, sequences, splits, args.models)
        cp.to_csv(args.output / 'cnn_predictions.csv', index=False)
        print('CNN-BiLSTM eligible test prefixes:', len(cp))
    if args.command in ['all', 'xu', 'train-xu']:
        xu_series = xu_input_series(series, args.xu_training_data) if args.xu_input in ['training', 'snapshot'] else series
        xp = predict_xu(xu_series, labels, mapping, splits, args.models, args.output, args.command == 'train-xu')
        xp.to_csv(args.output / 'xu_predictions.csv', index=False)
        print('STL-CNN-LSTM eligible tests:', len(xp))
    if args.command == 'all':
        evaluate(cp, xp, args.output)
    if args.command in ['all', 'field']:
        predict_field(field, args.models, args.output)
    print('Completed:', args.command)

def smooth(x, win=5):
    s = pd.Series(np.asarray(x, float))
    return s.rolling(win, center=True, min_periods=1).median().rolling(win, center=True, min_periods=1).mean().to_numpy(float)

def line(t, y):
    a = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(a, y, rcond=None)[0]
    yh = a @ np.array([m, b])
    sse = float(np.sum((y - yh) ** 2))
    return (float(m), float(b), sse, yh)

def best3(t, y, min_len=3):
    best = None
    for i in range(min_len, len(t) - 2 * min_len + 1):
        for j in range(i + min_len, len(t) - min_len + 1):
            f1 = line(t[:i], y[:i])
            f2 = line(t[i:j], y[i:j])
            f3 = line(t[j:], y[j:])
            total = f1[2] + f2[2] + f3[2]
            if best is None or total < best[0]:
                best = (total, i, j, f1, f2, f3)
    return best

def analyze(case, pid, t_all, rk_all, end=None):
    end = float(t_all[-1] if end is None else end)
    mask = (t_all >= START_MIN) & (t_all <= end)
    t = t_all[mask]
    raw = rk_all[mask]
    tr = smooth(raw, WIN)
    total, i, j, f1, f2, f3 = best3(t, tr, MIN_LEN)
    fit = np.concatenate([f1[3], f2[3], f3[3]])
    tss = float(np.sum((tr - np.mean(tr)) ** 2))
    r2 = 1 - total / tss if tss > 0 else np.nan
    summary = dict(condition=case, point_id=pid, start_min=float(t[0]), end_min=end, n_points=len(t), acc_end_min=float(t[i - 1]), fluc_start_min=float(t[i]), fluc_end_min=float(t[j - 1]), stab_start_b2_min=float(t[j]), total_sse=total, total_rmse=float(np.sqrt(total / len(t))), r2_piecewise=r2, seg1_n=i, seg1_slope=f1[0], seg1_intercept=f1[1], seg1_sse=f1[2], seg2_n=j - i, seg2_slope=f2[0], seg2_intercept=f2[1], seg2_sse=f2[2], seg3_n=len(t) - j, seg3_slope=f3[0], seg3_intercept=f3[1], seg3_sse=f3[2])
    detail = []
    for k, (tt, rr, yy, ff) in enumerate(zip(t, raw, tr, fit)):
        seg = 1 if k < i else 2 if k < j else 3
        res = float(yy - ff)
        detail.append(dict(condition=case, point_id=pid, time_min=float(tt), r_k=float(rr), smoothed_trend=float(yy), fitted_value=float(ff), segment=seg, residual=res, squared_error=res ** 2))
    return (summary, detail)
if __name__ == '__main__':
    main()