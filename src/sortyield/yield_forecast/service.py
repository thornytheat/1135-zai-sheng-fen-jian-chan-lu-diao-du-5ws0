from csv import DictReader
from datetime import date
from .contracts import RunHistory
from .engine import curves_for, model_vector, predict, select_methods


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = DictReader(source)
        return tuple(
            RunHistory(
                date.fromisoformat(row["handled"]),
                date.fromisoformat(row["closed"]),
                row["material"],
                model_vector(
                    int(row["kg"]),
                    float(row["impurity"]),
                    float(row["moisture"]),
                    row["line"],
                ),
                float(row["yield_rate"]),
                row["line"].strip(),
            )
            for row in rows
        )


def evidence(curves, forecasts, choices=()):
    choice_by_material = {choice.material: choice for choice in choices}
    return {
        "title": "产率拟合",
        "summary": f"拟合 {len(curves)} 个材质，{sum(row.rate is not None for row in forecasts)} 批有区间",
        "records": [
            {
                "料批": row.lot,
                "材质": row.fit_group,
                "产率": row.rate,
                "下界": row.lower,
                "上界": row.upper,
                "估法": row.method,
                "结果": row.notice,
            }
            for row in forecasts
        ],
        "拟合记录数": {
            key: {
                "估法": choice_by_material[key].method if key in choice_by_material else None,
                "训练": row.count,
                "校准": row.validation,
            }
            for key, row in curves.items()
        },
        "估法选择": [
            {
                "材质": choice.material,
                "估法": choice.method,
                "依据": choice.reason,
                "回测作业日": [day.isoformat() for day in choice.window],
                "在用产线": list(choice.lines),
                "回测": [
                    {
                        "估法": score.method,
                        "出料绝对误差和（kg）": score.error_kg,
                        "估算批数": score.estimates,
                        "漏估批数": score.misses,
                        "实产低于下界": dict(score.breaches),
                    }
                    for score in choice.scores
                ],
            }
            for choice in choices
        ],
    }
