import numpy as np

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

# Personal: user-adjustable lane position offset (LaneOffsetCm, positive = right).
#
# The driving model steers back toward where it wants to be, roughly like pure pursuit
# with a lookahead of v * LOOKAHEAD_TIME. A constant lateral accel bias of 2 * offset / T^2
# balances that restoring force at about `offset` metres from the model's line, independent
# of speed. The real shift depends on the model, so treat the cm value as approximate.
LANE_OFFSET_MAX_CM = 50
LANE_OFFSET_STEP_CM = 5
LOOKAHEAD_TIME = 2.0  # s
MAX_BIAS_LAT_ACCEL = 0.3  # m/s^2, hard cap on the bias
FADE_SPEEDS = [4.0, 9.0]  # m/s, no bias at parking-lot speeds, full by ~20 mph
FILTER_TC = 1.0  # s, eases taps and lane-change hand-offs in and out

# Personal: optional closed loop (LaneAutoCenter). Measures the car's position in the lane from the
# model's lane lines and slowly learns the lateral accel bias needed to sit at lane center + offset.
KI = 0.06  # (m/s^2) per m per s, ~8 s time constant at a 2 s lookahead
MAX_LEARNED = 0.25  # m/s^2
MAX_ERR = 1.0  # m, bigger errors are likely a mis-measurement or lane split
MIN_LINE_PROB = 0.7
MAX_LINE_STD = 0.25  # m
LANE_WIDTH_RANGE = (2.6, 4.6)  # m

LaneChangeState = log.LaneChangeState


def clip_offset_cm(offset_cm: int) -> int:
  return int(np.clip(offset_cm, -LANE_OFFSET_MAX_CM, LANE_OFFSET_MAX_CM))


def measure_lane_position(model_v2) -> float | None:
  """Car position relative to lane center in metres (positive = right of center), None if lane lines are unreliable."""
  lines, probs, stds = model_v2.laneLines, model_v2.laneLineProbs, model_v2.laneLineStds
  if len(lines) < 3 or len(probs) < 3:
    return None
  if probs[1] <= MIN_LINE_PROB or probs[2] <= MIN_LINE_PROB:
    return None
  if len(stds) >= 3 and (stds[1] >= MAX_LINE_STD or stds[2] >= MAX_LINE_STD):
    return None
  if len(lines[1].x) == 0 or len(lines[2].x) == 0:
    return None

  y_left = float(np.interp(0.0, lines[1].x, lines[1].y))
  y_right = float(np.interp(0.0, lines[2].x, lines[2].y))
  if not LANE_WIDTH_RANGE[0] <= y_right - y_left <= LANE_WIDTH_RANGE[1]:
    return None
  return -(y_left + y_right) / 2.0


class LaneOffset:
  def __init__(self):
    self.offset_cm = 0
    self.learned = 0.0  # m/s^2, kept across disengagements within a drive
    self.filter = FirstOrderFilter(0.0, FILTER_TC, DT_CTRL)

  def update_param(self, offset_cm) -> None:
    self.offset_cm = clip_offset_cm(offset_cm or 0)

  def update(self, active: bool, v_ego: float, lane_change_state, model_v2=None,
             steering_pressed: bool = False, auto_center: bool = False) -> float:
    """Returns the curvature bias to add to the model's desired curvature (openpilot convention, positive = right)."""
    if not active:
      self.filter.x = 0.0
      return 0.0

    target = 0.0
    if lane_change_state == LaneChangeState.off:
      if auto_center:
        self._learn(v_ego, model_v2, steering_pressed)
      target = 2.0 * (self.offset_cm / 100.0) / LOOKAHEAD_TIME ** 2
      if auto_center:
        target += self.learned
      target = float(np.clip(target, -MAX_BIAS_LAT_ACCEL, MAX_BIAS_LAT_ACCEL))
      target *= float(np.interp(v_ego, FADE_SPEEDS, [0.0, 1.0]))

    lat_accel_bias = self.filter.update(target)
    return lat_accel_bias / max(v_ego, 1.0) ** 2

  def _learn(self, v_ego: float, model_v2, steering_pressed: bool) -> None:
    if steering_pressed or v_ego < FADE_SPEEDS[1] or model_v2 is None:
      return
    pos = measure_lane_position(model_v2)
    if pos is None:
      return
    err = self.offset_cm / 100.0 - pos
    if abs(err) > MAX_ERR:
      return
    self.learned = float(np.clip(self.learned + KI * err * DT_CTRL, -MAX_LEARNED, MAX_LEARNED))
