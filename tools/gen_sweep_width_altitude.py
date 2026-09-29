"""W(H) sweep-width lookup generator — reuse the certified Ch1 machinery.

목적
----
플랫폼 고도 H 별로 소인폭 ``W(H) = ∫ P_D(x) dx`` 를 계산해 룩업 테이블(JSON)로
저장한다. 이 표는 GCS v0.5 에서 셀별 **고도효율 배율**
``e_alt(H) = W(H) / W(H0)`` 로 쓰이며, Stone-SPX 의 ``cell_scale`` 에 곱해져
셀마다 고도가 달라도 hazard 를 올바르게 반영한다 (고도는 SPX 결정변수가 아니라
위치의 결정함수이므로 최적성 인증은 그대로 유지된다).

핵심: 소인폭은 **연구 코드의 실제 함수** ``ch1_evaluation.sweep_width_table`` 을
그대로 호출해 계산한다. 값을 여기서 다시 구현하지 않는다 — 그래야 룩업이
인증된 hazard 모델과 비트 단위로 일치한다.

이 모델에서 고도는 **슬랜트 거리** ``R = hypot(H, offset)`` 를 통해서만 W 에
영향을 준다 (지지반폭 400 m 는 search camera + weave 기하라 고도 무관). 따라서 고도가
오르면 발자국 폭은 그대로이나 단위면적 P_D 가 떨어져 **W 는 단조 감소**한다.

실행 (CPP 저장소 루트에서, cpp_search 가 import 가능한 환경)::

    python3 tools/gen_sweep_width_altitude.py \
        --min 600 --max 2000 --step 50 --channel fused \
        --out config/sweep_width_altitude.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cpp_search.core.models import SensorSpec
from cpp_search.chapters.ch1_evaluation import sweep_width_table


def sweep_widths_at_altitude(
    altitude_m: float,
    *,
    base: SensorSpec | None = None,
    integration_steps: int = 256,
) -> dict[tuple[str, str], float]:
    """{(target_profile, channel_mode): W} at one altitude, via the Ch1 integral."""
    base = base or SensorSpec.sr_z50()
    sensor = replace(base, altitude_m=float(altitude_m))
    rows = sweep_width_table(sensor, integration_steps=integration_steps)
    return {
        (row["target_profile"], row["channel_mode"]): row["sweep_width_m"]
        for row in rows
    }


def build_lookup(
    altitudes_m: list[float],
    *,
    channel_mode: str = "fused",
    reference_altitude_m: float = 600.0,
    integration_steps: int = 256,
) -> dict:
    """Build the W(H) + altitude-efficiency lookup for every subject profile.

    ``altitude_efficiency[i] = W(altitudes_m[i]) / W(reference_altitude_m)``.
    Altitudes are returned sorted ascending so downstream ``numpy.interp`` works.
    """
    base = SensorSpec.sr_z50()
    altitudes = sorted(float(h) for h in altitudes_m)

    reference = sweep_widths_at_altitude(
        reference_altitude_m, base=base, integration_steps=integration_steps
    )
    profiles = sorted({p for (p, c) in reference if c == channel_mode})
    if not profiles:
        raise ValueError(f"no rows for channel_mode={channel_mode!r}")

    lookup: dict = {
        "channel_mode": channel_mode,
        "reference_altitude_m": float(reference_altitude_m),
        "integration_steps": integration_steps,
        "altitudes_m": altitudes,
        "profiles": {},
        "note": (
            "W(H) = int P_D(x) dx via cpp_search.chapters.ch1_evaluation."
            "sweep_width_table; altitude enters only through slant range so W "
            "is monotone decreasing in H. altitude_efficiency = W(H)/W(ref) is "
            "the per-cell multiplier for stone_spx cell_scale."
        ),
    }

    # Cache per-altitude computations (each call does all profiles/channels).
    per_altitude = {
        h: sweep_widths_at_altitude(
            h, base=base, integration_steps=integration_steps
        )
        for h in altitudes
    }

    for profile in profiles:
        w_ref = reference[(profile, channel_mode)]
        sweep = [per_altitude[h][(profile, channel_mode)] for h in altitudes]
        efficiency = [
            (w / w_ref) if w_ref > 0.0 else 0.0 for w in sweep
        ]
        lookup["profiles"][profile] = {
            "reference_sweep_width_m": w_ref,
            "sweep_width_m": sweep,
            "altitude_efficiency": efficiency,
        }
    return lookup


def _frange(minimum: float, maximum: float, step: float) -> list[float]:
    values: list[float] = []
    value = minimum
    # inclusive of maximum within a small epsilon
    while value <= maximum + 1e-9:
        values.append(round(value, 6))
        value += step
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a W(H) sweep-width lookup.")
    parser.add_argument("--min", type=float, default=600.0, help="min altitude (m)")
    parser.add_argument("--max", type=float, default=2000.0, help="max altitude (m)")
    parser.add_argument("--step", type=float, default=50.0, help="altitude step (m)")
    parser.add_argument("--channel", default="fused", choices=("eo", "ir", "fused"))
    parser.add_argument("--ref", type=float, default=600.0, help="reference altitude (m)")
    parser.add_argument("--steps", type=int, default=256, help="Simpson integration steps")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sweep_width_altitude.json"),
        help="output JSON path",
    )
    args = parser.parse_args(argv)

    altitudes = _frange(args.min, args.max, args.step)
    if args.ref not in altitudes:
        altitudes = sorted(set(altitudes) | {args.ref})

    lookup = build_lookup(
        altitudes,
        channel_mode=args.channel,
        reference_altitude_m=args.ref,
        integration_steps=args.steps,
    )
    args.out.write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Console preview.
    print(f"wrote {args.out}  (channel={args.channel}, ref={args.ref:.0f} m)")
    header_alts = [a for a in lookup["altitudes_m"] if a in (args.min, args.ref, (args.min + args.max) / 2, args.max)]
    for profile, block in lookup["profiles"].items():
        w_ref = block["reference_sweep_width_m"]
        print(f"\n[{profile}]  W({args.ref:.0f} m) = {w_ref:.2f} m")
        for h in header_alts:
            i = lookup["altitudes_m"].index(h)
            print(
                f"  H={h:7.0f} m   W={block['sweep_width_m'][i]:7.2f} m"
                f"   e_alt={block['altitude_efficiency'][i]:.4f}"
            )


if __name__ == "__main__":
    main()
