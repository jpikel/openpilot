from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter


DECEL_THRESHOLD = -0.30  # m/s^2
DECEL_RELEASE_THRESHOLD = -0.20  # m/s^2
RED_PREVIEW_TIME = 3.0  # seconds
AMBER_PREVIEW_TIME = 5.5  # seconds
PREVIEW_TIME_HYSTERESIS = 0.35  # seconds
MIN_PREVIEW_SPEED = 8.0  # m/s; suppress stop-and-go traffic
LEAD_PROB_THRESHOLD = 0.80
LEAD_STABLE_TIME = 0.50  # seconds
MATCHED_SPEED_TOLERANCE = 0.50  # m/s
LEAD_SPEED_FILTER_RC = 0.75  # seconds


class LeadPreviewState(StrEnum):
  HIDDEN = "hidden"
  GREEN = "green"
  AMBER = "amber"
  RED = "red"


class LeadPreviewSuppression(StrEnum):
  NONE = "none"
  DISENGAGED = "disengaged"
  NO_LEAD = "noLead"
  STOP_AND_GO = "stopAndGo"
  DRIVER_BRAKING = "driverBraking"
  LANE_CHANGE = "laneChange"
  FCW = "fcw"
  LOW_CONFIDENCE = "lowConfidence"
  UNSTABLE_LEAD = "unstableLead"
  ATTRIBUTION = "attribution"


@dataclass(frozen=True)
class LeadPreviewResult:
  state: LeadPreviewState = LeadPreviewState.HIDDEN
  suppression: LeadPreviewSuppression = LeadPreviewSuppression.NO_LEAD
  decel_predicted: bool = False
  predicted_decel_time: float = 0.0
  lead_index: int = 0
  ego_speed: float = 0.0
  ego_acceleration: float = 0.0
  lead_distance: float = 0.0
  relative_speed: float = 0.0
  lead_speed: float = 0.0
  lead_acceleration: float = 0.0
  lead_probability: float = 0.0
  track_stable: bool = False
  actual_decel_onset: bool = False


def first_decel_crossing(times, accelerations, threshold: float = DECEL_THRESHOLD) -> float | None:
  """Return the interpolated first time the planned acceleration reaches threshold."""
  times_array = np.asarray(times, dtype=float)
  accel_array = np.asarray(accelerations, dtype=float)
  length = min(len(times_array), len(accel_array))
  if length == 0:
    return None

  times_array = times_array[:length]
  accel_array = accel_array[:length]
  crossing_indices = np.flatnonzero(accel_array <= threshold)
  if not len(crossing_indices):
    return None

  index = int(crossing_indices[0])
  if index == 0 or accel_array[index - 1] <= threshold:
    return float(times_array[index])

  t0, t1 = times_array[index - 1 : index + 1]
  a0, a1 = accel_array[index - 1 : index + 1]
  if a1 == a0:
    return float(t1)

  crossing_fraction = np.clip((threshold - a0) / (a1 - a0), 0.0, 1.0)
  return float(t0 + crossing_fraction * (t1 - t0))


class _TrackedLead:
  def __init__(self, dt: float):
    self.dt = dt
    self._speed_filter = FirstOrderFilter(0.0, LEAD_SPEED_FILTER_RC, dt, initialized=False)
    self.reset()

  def reset(self) -> None:
    self.identity: tuple[str, int] | None = None
    self.stable_time = 0.0
    self.present = False
    self.confident = False
    self.distance = 0.0
    self.lateral_offset = 0.0
    self.relative_speed = 0.0
    self.speed = 0.0
    self.acceleration = 0.0
    self.probability = 0.0
    self._speed_filter.initialized = False

  @property
  def stable(self) -> bool:
    return self.present and self.confident and self.stable_time >= LEAD_STABLE_TIME

  def update(self, lead) -> None:
    if lead is None or not lead.present:
      self.reset()
      return

    probability = float(lead.modelProb)
    if probability < LEAD_PROB_THRESHOLD:
      self.reset()
      self.present = True
      self.probability = probability
      return

    track_id = int(lead.radarTrackId)
    identity = ("radar", track_id) if track_id >= 0 else ("vision", -1)
    continuous = identity == self.identity
    if continuous and track_id < 0:
      # Vision-only leads have no identity. Large frame-to-frame jumps are treated as a target switch.
      expected_distance_delta = abs(float(lead.vRel)) * self.dt
      continuous = (
        abs(float(lead.dRel) - self.distance) <= expected_distance_delta + 3.0
        and abs(float(lead.yRel) - self.lateral_offset) <= 1.5
        and abs(float(lead.vLeadK) - self.speed) <= 4.0
      )

    if not continuous:
      self.identity = identity
      self.stable_time = 0.0
      self._speed_filter.initialized = False

    self.present = True
    self.confident = True
    self.distance = float(lead.dRel)
    self.lateral_offset = float(lead.yRel)
    self.relative_speed = float(lead.vRel)
    self.speed = float(self._speed_filter.update(float(lead.vLeadK)))
    self.acceleration = float(lead.aLeadK)
    self.probability = probability
    self.stable_time += self.dt


class LeadPreview:
  """Stateful, informational lead-deceleration preview. It never feeds control outputs."""

  def __init__(self, dt: float):
    self._tracks = [_TrackedLead(dt), _TrackedLead(dt)]
    self._state = LeadPreviewState.HIDDEN
    self._selected_identity: tuple[str, int] | None = None
    self._actual_decelerating = False

  def update(
    self,
    *,
    engaged: bool,
    ego_speed: float,
    ego_acceleration: float,
    brake_pressed: bool,
    standstill: bool,
    lane_change_active: bool,
    fcw: bool,
    limiting_lead_index: int | None,
    plan_times,
    planned_accelerations,
    leads,
  ) -> LeadPreviewResult:
    for index, track in enumerate(self._tracks):
      track.update(leads[index] if index < len(leads) else None)

    actual_decel_onset = ego_acceleration <= DECEL_THRESHOLD and not self._actual_decelerating
    if ego_acceleration <= DECEL_THRESHOLD:
      self._actual_decelerating = True
    elif ego_acceleration > DECEL_RELEASE_THRESHOLD:
      self._actual_decelerating = False

    lead_index = limiting_lead_index if limiting_lead_index is not None else 0
    track = self._tracks[lead_index]
    decel_time = first_decel_crossing(plan_times, planned_accelerations)
    decel_predicted = decel_time is not None

    common = {
      "decel_predicted": decel_predicted,
      "predicted_decel_time": decel_time or 0.0,
      "lead_index": lead_index,
      "ego_speed": float(ego_speed),
      "ego_acceleration": float(ego_acceleration),
      "lead_distance": track.distance,
      "relative_speed": track.relative_speed,
      "lead_speed": track.speed,
      "lead_acceleration": track.acceleration,
      "lead_probability": track.probability,
      "track_stable": track.stable,
      "actual_decel_onset": actual_decel_onset,
    }

    suppression = self._suppression_reason(engaged, ego_speed, brake_pressed, standstill, lane_change_active, fcw, track)
    if suppression != LeadPreviewSuppression.NONE:
      self._state = LeadPreviewState.HIDDEN
      self._selected_identity = None
      return LeadPreviewResult(suppression=suppression, **common)

    if limiting_lead_index is not None and not track.stable:
      self._state = LeadPreviewState.HIDDEN
      self._selected_identity = None
      return LeadPreviewResult(suppression=LeadPreviewSuppression.ATTRIBUTION, **common)

    identity_changed = track.identity != self._selected_identity
    self._selected_identity = track.identity
    previous_state = LeadPreviewState.HIDDEN if identity_changed else self._state

    # A matched or faster lead is following state, even if the trajectory is still settling.
    if track.relative_speed >= -MATCHED_SPEED_TOLERANCE or limiting_lead_index is None or decel_time is None:
      state = LeadPreviewState.GREEN
    elif previous_state == LeadPreviewState.RED and decel_time <= RED_PREVIEW_TIME + PREVIEW_TIME_HYSTERESIS:
      state = LeadPreviewState.RED
    elif decel_time <= RED_PREVIEW_TIME:
      state = LeadPreviewState.RED
    elif previous_state == LeadPreviewState.AMBER and decel_time <= AMBER_PREVIEW_TIME + PREVIEW_TIME_HYSTERESIS:
      state = LeadPreviewState.AMBER
    elif decel_time <= AMBER_PREVIEW_TIME:
      state = LeadPreviewState.AMBER
    else:
      state = LeadPreviewState.GREEN

    self._state = state
    return LeadPreviewResult(state=state, suppression=LeadPreviewSuppression.NONE, **common)

  @staticmethod
  def _suppression_reason(
    engaged: bool, ego_speed: float, brake_pressed: bool, standstill: bool, lane_change_active: bool, fcw: bool, track: _TrackedLead
  ) -> LeadPreviewSuppression:
    if not engaged:
      return LeadPreviewSuppression.DISENGAGED
    if not track.present:
      return LeadPreviewSuppression.NO_LEAD
    if brake_pressed:
      return LeadPreviewSuppression.DRIVER_BRAKING
    if lane_change_active:
      return LeadPreviewSuppression.LANE_CHANGE
    if fcw:
      return LeadPreviewSuppression.FCW
    if standstill or ego_speed < MIN_PREVIEW_SPEED:
      return LeadPreviewSuppression.STOP_AND_GO
    if not track.confident:
      return LeadPreviewSuppression.LOW_CONFIDENCE
    if not track.stable:
      return LeadPreviewSuppression.UNSTABLE_LEAD
    return LeadPreviewSuppression.NONE
