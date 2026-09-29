"""고도효율 룩업 소비자 — GCS v0.5 측 (cpp_search 의존 없음, numpy 만).

``gen_sweep_width_altitude.py`` 가 만든 JSON 을 읽어, 임의 고도 H 의 소인폭 W(H)
와 고도효율 배율 ``e_alt(H) = W(H)/W(H0)`` 를 선형보간으로 준다. 셀별 고도 배열을
받아 Stone-SPX 의 ``cell_scale`` 에 곱할 배율 배열을 돌려주는 것이 핵심 용도다.

원칙: 고도는 위치의 결정함수이므로 SPX 결정변수가 아니다. 셀별 hazard 는 여전히
고정 상수이고, 최적성 인증(볼록 미탐지함수 + 접평면 컷 + gap)은 그대로 성립한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class AltitudeEfficiency:
    """One subject profile's W(H) curve + altitude-efficiency multiplier."""

    def __init__(self, lookup: dict, profile: str) -> None:
        if profile not in lookup["profiles"]:
            raise KeyError(
                f"profile {profile!r} not in lookup "
                f"({sorted(lookup['profiles'])})"
            )
        block = lookup["profiles"][profile]
        self.profile = profile
        self.channel_mode = lookup["channel_mode"]
        self.reference_altitude_m = float(lookup["reference_altitude_m"])
        self.reference_sweep_width_m = float(block["reference_sweep_width_m"])
        self._alt = np.asarray(lookup["altitudes_m"], dtype=float)
        self._eff = np.asarray(block["altitude_efficiency"], dtype=float)
        self._w = np.asarray(block["sweep_width_m"], dtype=float)
        if self._alt.size < 2:
            raise ValueError("lookup needs at least two altitudes")
        # numpy.interp requires ascending x; the generator sorts, but be safe.
        order = np.argsort(self._alt)
        self._alt, self._eff, self._w = (
            self._alt[order],
            self._eff[order],
            self._w[order],
        )

    @classmethod
    def from_json(cls, path: str | Path, profile: str) -> "AltitudeEfficiency":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, profile)

    def efficiency(self, altitude_m):
        """e_alt(H) = W(H)/W(H0). Beyond the table range, edge value is held
        (W(H) is monotone, so this over-estimates slightly above --max — extend
        the table if vehicles routinely fly higher)."""
        return np.interp(np.asarray(altitude_m, dtype=float), self._alt, self._eff)

    def sweep_width_m(self, altitude_m):
        """W(H) in metres, linearly interpolated."""
        return np.interp(np.asarray(altitude_m, dtype=float), self._alt, self._w)

    def cell_scale(self, cell_altitudes_m, base_scale=None):
        """Per-cell hazard multiplier for stone_spx ``cell_scale``.

        ``base_scale`` is the existing per-cell terrain-observability array
        (all 1.0 in synthetic terrain); pass it to compound, or omit for pure
        altitude efficiency.
        """
        efficiency = self.efficiency(cell_altitudes_m)
        if base_scale is None:
            return efficiency
        return np.asarray(base_scale, dtype=float) * efficiency


def assign_cell_altitudes(
    cell_ground_elevation_m,
    *,
    min_clearance_agl_m: float,
    altitude_floor_m: float = 600.0,
):
    """[지형추종/climb 가정] H(cell) = max(floor, ground_elevation + clearance).

    기체가 지형 위로 상승해 지면 위 일정 여유를 유지하는 모델. 지형이 높은 셀은
    센서-지면 높이가 커져 W 가 떨어지므로 ``AltitudeEfficiency.cell_scale`` 의
    e_alt 소프트 회피가 의미를 갖는다. (고정-MSL 가정이면 아래
    ``fixed_msl_cell_geometry`` 를 쓴다.)
    """
    ground = np.asarray(cell_ground_elevation_m, dtype=float)
    return np.maximum(altitude_floor_m, ground + float(min_clearance_agl_m))


def fixed_msl_cell_geometry(
    cell_ground_elevation_m,
    *,
    flight_msl_m: float = 600.0,
    min_clearance_agl_m: float = 120.0,
):
    """[고정-MSL 가정] 셀별 (센서-지면 높이 H_eff, no-fly 마스크).

    기체가 해발 ``flight_msl_m`` 구간을 일정하게 난다. 지형이 솟을수록 센서-지면
    높이가 줄어든다::

        H_eff(cell)  = flight_msl - ground_elevation(cell)
        no_fly(cell) = H_eff < min_clearance_agl        (지형이 비행고도에 근접)

    반환: ``(h_eff_m, no_fly_mask)``.

    - ``h_eff_m`` 은 ``AltitudeEfficiency.efficiency(h_eff_m)`` 로 넣어 e_alt 를
      얻지만, SR-Z50 은 이 구간(≈50~600 m)에서 W 가 거의 일정(e_alt≈1)이므로
      hazard 배율 효과는 미미하다.
    - **핵심은 ``no_fly_mask``** 다. True 인 셀은 격자/``_adjacency`` 에서 제거해
      경로가 우회하도록 한다. 그 셀에 belief 질량이 있으면 탐색 불가이므로
      달성 가능한 탐지확률에 천장이 생긴다(인증이 이를 정직하게 반영해야 함).
    """
    ground = np.asarray(cell_ground_elevation_m, dtype=float)
    h_eff = float(flight_msl_m) - ground
    no_fly = h_eff < float(min_clearance_agl_m)
    return h_eff, no_fly


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe a W(H) lookup.")
    parser.add_argument("lookup", type=Path)
    parser.add_argument("--profile", default="MissingPerson-adult-nominal")
    parser.add_argument("--altitudes", type=float, nargs="+", default=[600, 900, 1200, 1500, 2000])
    args = parser.parse_args()

    curve = AltitudeEfficiency.from_json(args.lookup, args.profile)
    print(f"profile={curve.profile}  channel={curve.channel_mode}  "
          f"W({curve.reference_altitude_m:.0f} m)={curve.reference_sweep_width_m:.2f} m")
    for h in args.altitudes:
        print(f"  H={h:7.0f} m   W={float(curve.sweep_width_m(h)):7.2f} m   "
              f"e_alt={float(curve.efficiency(h)):.4f}")
