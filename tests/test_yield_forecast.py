from datetime import date, timedelta
from types import SimpleNamespace
from dataclasses import replace
import pytest
from sortyield.yield_forecast.contracts import (
    METHOD_PER_POLYMER,
    METHOD_SHARED,
    BacktestScore,
    RunHistory,
)
from sortyield.yield_forecast.engine import (
    backtest,
    backtest_window,
    center_fit,
    curves_for,
    decide,
    fit_curve,
    model_vector,
    per_polymer,
    predict,
    select_methods,
    selection_curves,
)

DAY = date(2026, 11, 12)


def sample():
    return [
        RunHistory(
            DAY + timedelta(days=i),
            DAY + timedelta(days=i + 1),
            "PP",
            model_vector(1000 + i * 9, 0.13, 0.04, "光选线"),
            0.5 + i * 0.02,
        )
        for i in range(12)
    ]


def lot(kg):
    return SimpleNamespace(lot="P111", material="PP", line="光选线", kg=kg)


def audit():
    return SimpleNamespace(approved_samples=True, impurity=0.13, moisture=0.04)


def test_cutoff_discards_pending_labels():
    # TRAP-42
    fold = DAY + timedelta(days=8)
    cut = DAY + timedelta(days=12)
    original = per_polymer(sample(), fold, cut)
    assert (
        per_polymer(
            [
                *sample(),
                replace(sample()[-1], closed=cut + timedelta(days=2), yield_rate=0),
            ],
            fold,
            cut,
        )
        == original
    )


def test_raw_scale_constant_columns_do_not_divide_by_zero():
    # TRAP-32
    point = center_fit(sample())
    assert min(point[1]) == 1e-6 and point[1][0] > 0


def test_per_lot_features_are_independent_of_batch_size():
    # TRAP-53
    curve = fit_curve("PP", sample()[:8], sample()[8:])
    a = predict(lot(1050), audit(), curve)
    assert [predict(lot(size), audit(), curve) for size in (1000, 1050, 1007)][1] == a


def test_out_of_support_skips_false_precision():
    # TRAP-44
    curve = fit_curve("PP", sample()[:8], sample()[8:])
    assert predict(lot(1000000), audit(), curve).rate is None


def test_interval_has_consistent_endpoints():
    curve = fit_curve("PP", sample()[:8], sample()[8:])
    estimate = predict(lot(1050), audit(), curve)
    assert 0 <= estimate.lower <= estimate.rate <= estimate.upper <= 1


def test_known_feature_slope_has_correct_direction():
    curve = fit_curve("PP", sample()[:8], sample()[8:])
    assert (
        predict(lot(1030), audit(), curve).rate
        > predict(lot(1002), audit(), curve).rate
    )


def test_insufficient_records_reject_fit():
    with pytest.raises(ValueError):
        fit_curve("PP", sample()[:2], [])


def test_no_valid_sampling_skips_curve_execution():
    assert predict(lot(1050), audit(), None).lower is None


def history():
    first = date(2026, 7, 1)
    rows = []

    def add(offset, material, kg, impurity, moisture, rate, line="光选线"):
        rows.append(
            RunHistory(
                first + timedelta(days=offset),
                first + timedelta(days=offset + 2),
                material,
                model_vector(kg, impurity, moisture, line),
                rate,
                line,
            )
        )

    def features(index):
        return 1000 + 10 * (index % 26), 0.05 + 0.005 * (index % 10), 0.04

    for index in range(40):
        kg, impurity, moisture = features(index * 3 + 1)
        add(index % 26, "PET", kg, impurity, moisture, 0.9 - 1.5 * impurity + 0.02 * moisture)
    for index in range(30):
        kg, impurity, moisture = features(index * 2 + 1)
        noise = -0.09 if index % 2 else 0.09
        add(index % 26, "PP", kg, impurity, moisture, 0.9 - 1.5 * impurity + noise)
    for index in range(26, 43):
        kg, impurity, moisture = features(index)
        clean = 0.9 - 1.5 * impurity + 0.02 * moisture
        add(index, "PET", kg, impurity, moisture, clean)
        add(index, "PP", kg, impurity, moisture, clean if index >= 38 else clean + (0.09 if index % 2 else -0.09))
    return rows


def choices_for(rows, in_use=None, days=5):
    return select_methods(
        rows,
        date(2026, 7, 21),
        date(2026, 8, 20),
        in_use or {"PET": {"光选线"}, "PP": {"光选线"}},
        days=days,
    )


def test_backtest_window_uses_last_five_closed_workdays():
    rows = history()
    window = backtest_window(rows, date(2026, 8, 20), 5)
    assert window == tuple(date(2026, 7, 1) + timedelta(days=day) for day in range(38, 43))
    closed_early = backtest_window(rows, date(2026, 7, 4), 5)
    assert closed_early == tuple(date(2026, 7, 1) + timedelta(days=day) for day in range(2))
    assert backtest_window(rows, date(2026, 7, 1), 5) == ()


def test_fit_and_calibration_exclude_backtest_records():
    rows = history()
    fold = date(2026, 7, 21)
    window = backtest_window(rows, date(2026, 8, 20), 5)
    start = window[0]
    per, pooled = selection_curves(rows, fold, start)
    prior = [row for row in rows if row.handled < start]

    def training(material):
        return sum(
            row.polymer == material and row.handled < fold and row.closed <= fold
            for row in prior
        )

    def calibration(material):
        return sum(
            row.polymer == material
            and row.handled >= fold
            and row.closed <= start - timedelta(days=1)
            for row in prior
        )

    assert per["PET"].count == training("PET")
    assert per["PP"].count == training("PP")
    assert per["PET"].validation == calibration("PET")
    assert per["PP"].validation == calibration("PP")
    assert pooled.count == training("PET") + training("PP")
    assert pooled.validation == calibration("PET") + calibration("PP")
    assert training("PET") + training("PP") + sum(
        calibration(material) for material in ("PET", "PP")
    ) < len(rows)


def test_backtest_scores_only_rows_closed_before_the_shift_cutoff():
    rows = list(history())
    cutoff = date(2026, 8, 10)
    window = backtest_window(rows, cutoff, 5)
    pending = replace(
        rows[0],
        handled=window[-1],
        closed=cutoff + timedelta(days=1),
    )
    rows.append(pending)
    per, pooled = selection_curves(rows, date(2026, 7, 21), window[0])
    within = [
        row
        for row in rows
        if window[0] <= row.handled <= window[-1] and row.closed <= cutoff
    ]
    assert pending not in within
    for material in ("PET", "PP"):
        scored = backtest(
            [row for row in within if row.polymer == material],
            per[material],
            METHOD_PER_POLYMER,
        )
        assert scored.estimates == sum(row.polymer == material for row in within)
    assert pooled is not None


def test_shared_history_wins_when_kg_error_drops():
    choice = choices_for(history())["PP"]
    assert choice.method == METHOD_SHARED
    assert choice.reason == "出料绝对误差之和下降"
    baseline, candidate = choice.scores
    assert candidate.error_kg < baseline.error_kg
    assert baseline.misses == candidate.misses == 0


def test_shared_history_is_kept_only_when_error_strictly_drops():
    baseline = BacktestScore(METHOD_PER_POLYMER, 10.0, 5, 0, ())
    candidate = BacktestScore(METHOD_SHARED, 10.0, 5, 0, ())
    assert decide(baseline, candidate, ("光选线",), {"光选线"}) == (
        METHOD_PER_POLYMER,
        "出料绝对误差之和未下降",
    )
    better = BacktestScore(METHOD_SHARED, 9.999, 5, 0, ())
    assert decide(baseline, better, ("光选线",), {"光选线"})[0] == METHOD_SHARED


def test_missing_backtest_estimate_blocks_the_switch():
    baseline = BacktestScore(METHOD_PER_POLYMER, 10.0, 5, 0, ())
    candidate = BacktestScore(METHOD_SHARED, 1.0, 4, 1, ())
    assert decide(baseline, candidate, ("光选线",), {"光选线"}) == (
        METHOD_PER_POLYMER,
        "估法漏估回测批",
    )
    assert decide(candidate, baseline, ("光选线",), {"光选线"})[1] == "估法漏估回测批"


def test_line_without_backtest_sample_keeps_the_original_method():
    rows = [row for row in history() if row.line == "光选线"]
    choice = choices_for(rows, {"PET": {"光选线", "涡电流线"}})["PET"]
    assert choice.method == METHOD_PER_POLYMER
    assert choice.reason == "在用产线缺回测样本"
    assert choice.lines == ("光选线", "涡电流线")


def test_more_lower_bound_breaches_on_a_line_blocks_the_switch():
    baseline = BacktestScore(METHOD_PER_POLYMER, 20.0, 5, 0, (("涡电流线", 1),))
    candidate = BacktestScore(METHOD_SHARED, 5.0, 5, 0, (("涡电流线", 2),))
    assert decide(baseline, candidate, ("光选线", "涡电流线"), {"光选线", "涡电流线"}) == (
        METHOD_PER_POLYMER,
        "产线下界失守批次增加",
    )
    equal = BacktestScore(METHOD_SHARED, 5.0, 5, 0, (("涡电流线", 1),))
    assert decide(baseline, equal, ("涡电流线",), {"涡电流线"})[0] == METHOD_SHARED


def test_selected_method_drives_forecast_lower_bound_and_reserve():
    rows = history()
    choices = choices_for(rows)
    assert choices["PP"].method == METHOD_SHARED
    lot = SimpleNamespace(lot="P1", material="PP", line="光选线", kg=1003)
    audit = SimpleNamespace(approved_samples=True, impurity=0.07, moisture=0.04)
    fold, cutoff = date(2026, 7, 21), date(2026, 8, 20)
    per = per_polymer(rows, fold, cutoff)
    curves = curves_for(rows, fold, cutoff, choices)
    shared = predict(lot, audit, curves["PP"], METHOD_SHARED)
    separate = predict(lot, audit, per["PP"], METHOD_PER_POLYMER)
    assert shared.method == METHOD_SHARED and separate.method == METHOD_PER_POLYMER
    assert curves["PP"] is not per["PP"]
    assert shared.lower != separate.lower or shared.rate != separate.rate


def test_switch_never_leaks_into_materials_left_on_the_original_method():
    rows = history()
    fold, cutoff = date(2026, 7, 21), date(2026, 8, 20)
    choices = dict(choices_for(rows))
    per = per_polymer(rows, fold, cutoff)
    choices["PET"] = replace(
        choices["PET"], method=METHOD_PER_POLYMER, reason="出料绝对误差之和未下降"
    )
    curves = curves_for(rows, fold, cutoff, choices)
    assert curves["PET"] == per["PET"]
    assert curves["PP"] is not per["PP"]
