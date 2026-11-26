from sortyield.pipeline import run


def test_batch_reports_damaged_samples_and_capacity_tradeoff():
    limited, expanded = run(1), run(100)
    assert len(limited["rows"]) == 18 and len(limited["stages"]) == 6
    assert limited["metrics"]["预留（kg）"] < expanded["metrics"]["预留（kg）"]
    assert limited["stages"][1]["未匹配样本"] == ("T-orphan",)
    for record in expanded["stages"][-1]["records"]:
        assert (
            record["来料（kg）"]
            == record["预留（kg）"]
            + record["未分配预计出料（kg）"]
            + record["预计残渣（kg）"]
        )


def test_all_scenarios_keep_stage_one_and_training_evidence_identical():
    candidates = [run(limit) for limit in (1, 6, 18, 35)]
    reference = candidates[0]["stages"][:4]
    assert all(item["stages"][:4] == reference for item in candidates)
    for candidate in candidates:
        allocation = candidate["stages"][4]
        comparisons = allocation["班次额度试算"]
        assert (
            sum(allocation["仓位预留"].values()) == candidate["metrics"]["预留（kg）"]
        )
        assert any(
            row["预留（kg）"] == candidate["metrics"]["预留（kg）"]
            for row in comparisons
        )


def test_max_capacity_respects_physical_bunkers_before_quota():
    expanded = run(100)
    reservations = expanded["stages"][4]
    final = expanded["stages"][5]
    assert (
        sum(reservations["仓位剩余"].values()) == expanded["metrics"]["可用仓容（kg）"]
    )
    assert reservations["仓位剩余"] == final["已对账仓位"]
    assert all(amount >= 0 for amount in final["已对账仓位"].values())
    assert all(
        row["班次上限（kg）"] >= row["预留（kg）"]
        for row in reservations["班次额度试算"]
    )


def test_held_batches_remain_in_closing_mass_reconciliation():
    result = run(1)
    totals = result["stages"][5]["records"]
    pending = [row for row in result["rows"] if row["bunker"] is None]
    assert pending
    assert {row["lot"] for row in pending} <= {row["料批"] for row in totals}
    assert sum(row["来料（kg）"] for row in totals) == result["metrics"]["来料（kg）"]


def test_shift_forecast_uses_the_selected_method_per_material():
    from sortyield.yield_forecast.engine import (
        curves_for,
        per_polymer,
        select_methods,
    )
    from sortyield.yield_forecast.service import read
    from sortyield.pipeline import CUT, DATA, FOLD
    from sortyield.weighbridge import service as weigh

    result = run()
    journal = weigh.read(DATA / "weighbridge_shift_north.csv", CUT)
    history = read(DATA / "separated_material_yield_history.csv")
    in_use = {}
    for lot in journal.lots:
        in_use.setdefault(lot.material, set()).add(lot.line)
    choices = select_methods(history, FOLD, CUT.date(), in_use)
    curves = curves_for(history, FOLD, CUT.date(), choices)
    separate = per_polymer(history, FOLD, CUT.date())
    evidence = result["stages"][2]
    reported = {row["材质"]: row for row in evidence["估法选择"]}
    assert set(reported) == set(choices)
    forecast_rows = {row["料批"]: row for row in evidence["records"]}
    for choice in choices.values():
        assert reported[choice.material]["估法"] == choice.method
        assert reported[choice.material]["依据"] == choice.reason
        if choice.method == "共用历史":
            assert curves[choice.material] is not separate[choice.material]
        else:
            assert curves[choice.material] == separate.get(choice.material)
    for row in result["rows"]:
        method = forecast_rows[row["lot"]]["估法"]
        assert method == choices[row["material"]].method
    lower = {
        record["料批"]: record["下界"] for record in evidence["records"]
    }
    for grade in result["stages"][3]["records"]:
        rate = lower[grade["料批"]]
        if rate is not None and grade["等级"] == "清洁颗粒":
            assert grade["保守排仓（kg）"] <= grade["预测出料（kg）"]
