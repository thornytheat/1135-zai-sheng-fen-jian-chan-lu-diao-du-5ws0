from collections import defaultdict
from datetime import timedelta
from math import ceil, expm1, fsum, log1p, sqrt
from .contracts import (
    METHOD_PER_POLYMER,
    METHOD_SHARED,
    BacktestScore,
    Curve,
    Estimate,
    MethodChoice,
)


def model_vector(kg, impurity, moisture, line):
    return log1p(kg), impurity, moisture, float(line == "涡电流线")


def observation_kg(row):
    return round(expm1(row.features[0]))


def line_of(row):
    if row.line:
        return row.line
    return "涡电流线" if row.features[3] == 1 else "光选线"


def normalize(raw, center, scale):
    return tuple(
        (x - mu) / sigma for x, mu, sigma in zip(raw, center, scale, strict=True)
    )


def center_fit(rows):
    columns = tuple(zip(*(row.features for row in rows), strict=True))
    center = tuple(fsum(col) / len(col) for col in columns)
    scale = tuple(
        max(1e-6, sqrt(fsum((x - mu) ** 2 for x in col) / len(col)))
        for col, mu in zip(columns, center, strict=True)
    )
    return center, scale, tuple((min(col), max(col)) for col in columns)


def clipped_prediction(vector, center, scale, params, base):
    weighted = fsum(
        w * x for w, x in zip(params, normalize(vector, center, scale), strict=True)
    )
    return max(0, min(1, base + weighted))


def fit_curve(material, train, validation, penalty=0.10):
    if len(train) < 5:
        raise ValueError("回归样本不足")
    center, scale, limits = center_fit(train)
    values = [normalize(row.features, center, scale) for row in train]
    target = [row.yield_rate for row in train]
    base = fsum(target) / len(target)
    params = [0.0] * len(center)
    for _ in range(150):
        errors = [
            base + fsum(w * x for w, x in zip(params, features, strict=True)) - y
            for features, y in zip(values, target, strict=True)
        ]
        gradient = [
            fsum(
                error * features[i]
                for error, features in zip(errors, values, strict=True)
            )
            / len(train)
            + penalty * params[i]
            for i in range(len(params))
        ]
        params = [
            weight - 0.1 * slope for weight, slope in zip(params, gradient, strict=True)
        ]
    residual = sorted(
        abs(
            row.yield_rate
            - clipped_prediction(row.features, center, scale, params, base)
        )
        for row in validation
    )
    quantile = (
        residual[min(len(residual), ceil(0.9 * (len(residual) + 1))) - 1]
        if residual
        else 0.12
    )
    return Curve(
        material,
        center,
        scale,
        limits,
        tuple(params),
        base,
        quantile,
        len(train),
        len(validation),
    )


def _split(history, fold, cutoff):
    train = defaultdict(list)
    calibration = defaultdict(list)
    for row in history:
        if row.handled < fold and row.closed <= fold:
            train[row.polymer].append(row)
        elif row.handled >= fold and row.closed <= cutoff:
            calibration[row.polymer].append(row)
    return train, calibration


def per_polymer(history, fold, cutoff):
    train, calibration = _split(history, fold, cutoff)
    curves = {
        material: fit_curve(material, rows, calibration[material])
        for material, rows in sorted(train.items())
        if len(rows) >= 5
    }
    return curves


def shared_history(history, fold, cutoff):
    train, calibration = _split(history, fold, cutoff)
    pooled_train = [row for rows in train.values() for row in rows]
    pooled_calibration = [row for rows in calibration.values() for row in rows]
    if len(pooled_train) < 5:
        return None
    return fit_curve(METHOD_SHARED, pooled_train, pooled_calibration)


def backtest_window(history, cutoff, days=5):
    closed_days = sorted({row.handled for row in history if row.closed <= cutoff})
    return tuple(closed_days[-days:])


def _project(features, curve):
    if curve is None:
        return None
    distance = max(
        max(low - x, x - high, 0) / sd
        for x, (low, high), sd in zip(features, curve.limits, curve.scale, strict=True)
    )
    if distance > 2.5:
        return None
    score = clipped_prediction(
        features, curve.center, curve.scale, curve.params, curve.rate
    )
    return score, max(0, score - curve.loss_radius)


def backtest(rows, curve, method):
    error_kg = 0.0
    estimates = 0
    misses = 0
    breaches = defaultdict(int)
    for row in rows:
        projected = _project(row.features, curve)
        if projected is None:
            misses += 1
            continue
        score, lower = projected
        estimates += 1
        error_kg += observation_kg(row) * abs(row.yield_rate - score)
        if row.yield_rate < lower:
            breaches[line_of(row)] += 1
    return BacktestScore(
        method, round(error_kg, 8), estimates, misses, tuple(sorted(breaches.items()))
    )


def selection_curves(history, fold, start):
    """Fit both candidate methods using only records handled before the window."""
    prior = [row for row in history if row.handled < start]
    fit_cutoff = start - timedelta(days=1)
    return per_polymer(prior, fold, fit_cutoff), shared_history(prior, fold, fit_cutoff)


def decide(baseline, candidate, lines, covered):
    if any(line not in covered for line in lines):
        return METHOD_PER_POLYMER, "在用产线缺回测样本"
    if baseline.misses or candidate.misses:
        return METHOD_PER_POLYMER, "估法漏估回测批"
    if any(candidate.breach(line) > baseline.breach(line) for line in lines):
        return METHOD_PER_POLYMER, "产线下界失守批次增加"
    if candidate.error_kg < baseline.error_kg:
        return METHOD_SHARED, "出料绝对误差之和下降"
    return METHOD_PER_POLYMER, "出料绝对误差之和未下降"


def select_methods(history, fold, cutoff, in_use, days=5):
    window = backtest_window(history, cutoff, days)
    if not window:
        return {}
    start = window[0]
    per, pooled = selection_curves(history, fold, start)
    rows = [
        row
        for row in history
        if start <= row.handled <= window[-1] and row.closed <= cutoff
    ]
    choices = {}
    for material in sorted(set(in_use)):
        material_rows = [row for row in rows if row.polymer == material]
        lines = tuple(sorted(in_use[material]))
        baseline = backtest(material_rows, per.get(material), METHOD_PER_POLYMER)
        candidate = backtest(material_rows, pooled, METHOD_SHARED)
        covered = {line_of(row) for row in material_rows}
        method, reason = decide(baseline, candidate, lines, covered)
        choices[material] = MethodChoice(
            material,
            method,
            reason,
            window,
            lines,
            (baseline, candidate),
        )
    return choices


def curves_for(history, fold, cutoff, choices):
    per = per_polymer(history, fold, cutoff)
    pooled = shared_history(history, fold, cutoff)
    curves = dict(per)
    for material, choice in choices.items():
        if choice.method == METHOD_SHARED and pooled is not None:
            curves[material] = pooled
    return curves


def predict(lot, audit, curve, method=None):
    if curve is None or not audit.approved_samples:
        return Estimate(
            lot.lot, None, None, None, lot.material, "缺足量抽检或拟合记录", method
        )
    vector = model_vector(lot.kg, audit.impurity, audit.moisture, lot.line)
    projected = _project(vector, curve)
    if projected is None:
        return Estimate(
            lot.lot, None, None, None, lot.material, "料批超出训练支撑", method
        )
    score, lower = projected
    return Estimate(
        lot.lot,
        score,
        lower,
        min(1, score + curve.loss_radius),
        lot.material,
        "已估算",
        method,
    )
