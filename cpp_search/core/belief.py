"""로그오즈 증거지도와 POC / POD / POS.

증거지도와 배분지도는 **다른 데이터**다. 이 파일이 그 구분을 강제한다.

핵심 수식
---------
* 로그오즈 (``_logit`` / ``_sigmoid``)

      l = log( p / (1-p) ),      p = 1 / (1 + exp(-l))

  ``_sigmoid``는 l의 부호에 따라 식을 바꿔 오버플로를 피한다.

* 이진 센서 보고의 베이즈 갱신 (``LogOddsEvidenceGrid.update``)

      탐지 보고    : l <- l + log( PD / PF )
      미탐지 보고  : l <- l + log( (1-PD) / (1-PF) )

  Ch0 config의 ``detection.reference_detection_probability`` = PD,
  ``detection.false_alarm_probability`` = PF가 이 두 상수를 만든다.
  (PD 0.86 / PF 0.05 -> +2.845 / -1.915)

* 두 지도의 분리

      점유 확률       occupancy    = sigmoid(l)          (셀마다 독립)
      표적 위치 확률   location     = occupancy / sum(occupancy)

  전자는 "여기 뭔가 있나", 후자는 "단일 표적이 여기 있을 확률"이다.
  노력배분은 반드시 후자를 쓴다.

* 탐색 성공확률 (``SearchProbabilityMetrics.from_cell_probabilities``)

      POC = sum_{탐색한 셀} p(x)
      POS = sum_x p(x) * PD(x)
      POD = POS / POC

  POS는 확률가중 탐지확률이지 "POC 곱하기 평균 PD"가 아니다.

의존
----
* 위: numpy만. 도메인 객체에 의존하지 않는 순수 확률 계산.
* 아래: Ch0의 증거 계약 출력, ``teamwork/asoc``의 증거 증분 개념.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable, Sequence

import numpy as np


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    positive = log_odds >= 0.0
    result = np.empty_like(log_odds, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    negative_exp = np.exp(log_odds[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


@dataclass(slots=True)
class LogOddsEvidenceGrid:
    """Occupancy evidence grid with a normalized single-target location view."""

    log_odds: np.ndarray

    @classmethod
    def from_prior(cls, prior_probability: Sequence[float] | np.ndarray) -> "LogOddsEvidenceGrid":
        prior = np.asarray(prior_probability, dtype=float)
        if prior.size == 0:
            raise ValueError("prior_probability must not be empty")
        if np.any(prior < 0.0) or not np.isfinite(prior).all():
            raise ValueError("prior_probability must be finite and non-negative")
        total = float(prior.sum())
        if total <= 0.0:
            raise ValueError("prior_probability must contain positive mass")
        normalized = prior / total
        return cls(_logit(normalized))

    @property
    def occupancy_probability(self) -> np.ndarray:
        return _sigmoid(self.log_odds)

    @property
    def target_location_probability(self) -> np.ndarray:
        occupancy = self.occupancy_probability
        total = float(occupancy.sum())
        if total <= 0.0:
            return np.full(occupancy.shape, 1.0 / occupancy.size)
        return occupancy / total

    def update(
        self,
        observed: np.ndarray,
        *,
        detected: bool,
        detection_probability: float | np.ndarray,
        false_alarm_probability: float | np.ndarray,
    ) -> None:
        """Apply one binary sensor report to selected grid cells."""

        mask = np.asarray(observed, dtype=bool)
        if mask.shape != self.log_odds.shape:
            raise ValueError("observed mask must match the grid shape")
        pd = np.broadcast_to(np.asarray(detection_probability, dtype=float), self.log_odds.shape)
        pf = np.broadcast_to(np.asarray(false_alarm_probability, dtype=float), self.log_odds.shape)
        if np.any((pd <= 0.0) | (pd >= 1.0)):
            raise ValueError("detection_probability must be in (0, 1)")
        if np.any((pf <= 0.0) | (pf >= 1.0)):
            raise ValueError("false_alarm_probability must be in (0, 1)")
        # 베이즈 로그오즈 갱신. 사전 로그오즈에 우도비의 로그를 더한다.
        #   탐지   : + ln( PD / PF )
        #   미탐지 : + ln( (1-PD) / (1-PF) )   (PD > PF이므로 음수)
        if detected:
            increment = np.log(pd / pf)
        else:
            increment = np.log((1.0 - pd) / (1.0 - pf))
        self.log_odds[mask] += increment[mask]


@dataclass(frozen=True, slots=True)
class SearchProbabilityMetrics:
    """POC, POD, and POS for a discrete single-target probability map."""

    probability_of_containment: float
    probability_of_detection_given_containment: float
    probability_of_success: float

    @classmethod
    def from_cell_probabilities(
        cls,
        target_location_probability: Iterable[float],
        cell_detection_probability: Iterable[float],
    ) -> "SearchProbabilityMetrics":
        location = np.asarray(tuple(target_location_probability), dtype=float)
        detection = np.asarray(tuple(cell_detection_probability), dtype=float)
        if location.shape != detection.shape or location.size == 0:
            raise ValueError("location and detection arrays must have the same non-zero shape")
        if np.any(location < 0.0) or not np.isfinite(location).all():
            raise ValueError("location probabilities must be finite and non-negative")
        if np.any((detection < 0.0) | (detection > 1.0)):
            raise ValueError("detection probabilities must be in [0, 1]")
        total = float(location.sum())
        if total <= 0.0:
            raise ValueError("location probabilities must contain positive mass")
        normalized = location / total
        searched = detection > 0.0
        # POC = 탐색한 셀들의 표적 존재확률 합
        # POS = sum_x p(x) * PD(x)   <- 확률가중 탐지확률 (POC * 평균PD 아님)
        # POD = POS / POC            <- 탐색한 영역 안에서의 조건부 탐지확률
        poc = float(normalized[searched].sum())
        pos = float(np.dot(normalized, detection))
        pod = pos / poc if poc > 0.0 else 0.0
        return cls(poc, pod, pos)
