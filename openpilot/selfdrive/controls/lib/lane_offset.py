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

LaneChangeState = log.LaneChangeState


def clip_offset_cm(offset_cm: int) -> int:
  return int(np.clip(offset_cm, -LANE_OFFSET_MAX_CM, LANE_OFFSET_MAX_CM))


class LaneOffset:
  def __init__(self):
    self.offset_cm = 0
    self.filter = FirstOrderFilter(0.0, FILTER_TC, DT_CTRL)

  def update_param(self, offset_cm) -> None:
    self.offset_cm = clip_offset_cm(offset_cm or 0)

  def update(self, active: bool, v_ego: float, lane_change_state) -> float:
    """Returns the curvature bias to add to the model's desired curvature (openpilot convention, positive = right)."""
    if not active:
      self.filter.x = 0.0
      return 0.0

    target = 0.0
    if lane_change_state == LaneChangeState.off:
      target = 2.0 * (self.offset_cm / 100.0) / LOOKAHEAD_TIME ** 2
      target = float(np.clip(target, -MAX_BIAS_LAT_ACCEL, MAX_BIAS_LAT_ACCEL))
      target *= float(np.interp(v_ego, FADE_SPEEDS, [0.0, 1.0]))

    lat_accel_bias = self.filter.update(target)
    return lat_accel_bias / max(v_ego, 1.0) ** 2
