from dataclasses import dataclass
from datetime import date
from math import isfinite

METHOD_PER_POLYMER = "材质独立"
METHOD_SHARED = "共用历史"


@dataclass(frozen=True)
class RunHistory:
    handled: date
    closed: date
    polymer: str
    features: tuple[float, ...]
    yield_rate: float
    line: str = ""

    def __post_init__(self):
        if self.closed <= self.handled or not self.polymer or len(self.features) != 4:
            raise ValueError("产率样本时序、品种或维度错误")
        if not all(isfinite(x) for x in self.features) or not 0 <= self.yield_rate <= 1:
            raise ValueError("产率观测值越界")


@dataclass(frozen=True)
class Curve:
    material: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    limits: tuple[tuple[float, float], ...]
    params: tuple[float, ...]
    rate: float
    loss_radius: float
    count: int
    validation: int


@dataclass(frozen=True)
class Estimate:
    lot: str
    rate: float | None
    lower: float | None
    upper: float | None
    fit_group: str
    notice: str
    method: str | None = None


@dataclass(frozen=True)
class BacktestScore:
    method: str
    error_kg: float
    estimates: int
    misses: int
    breaches: tuple[tuple[str, int], ...]

    def breach(self, line):
        return dict(self.breaches).get(line, 0)


@dataclass(frozen=True)
class MethodChoice:
    material: str
    method: str
    reason: str
    window: tuple[date, ...]
    lines: tuple[str, ...]
    scores: tuple[BacktestScore, ...]
