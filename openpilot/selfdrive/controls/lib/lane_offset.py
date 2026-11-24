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
LANE_WIDTH_RANGE = (2.6, 4.2)  # m
MAX_WIDTH_CHANGE = 0.25  # m vs the recent average, a lane forming or splitting (exits, merges) moves the measured center
WIDTH_FILTER_TC = 5.0  # s

# Personal: curve term of the closed loop. The car sits on the inside of curves (measured on the XT6: ~0.2-0.4 m at
# 1-1.5 m/s^2, both directions). A slow integrator can't fix that: it winds up during the curve and then pushes the
# car over the opposite line once the road straightens. So the slow term above only learns on near-straight road,
# and curves get an outward bias proportional to the model's own lateral accel request. It is zero as soon as the
# curve ends, and the gain is learned from the lane position measured in curves.
STRAIGHT_LAT_ACCEL = 0.4  # m/s^2, the slow term learns below this
CURVE_LAT_ACCEL = 0.6  # m/s^2, the curve gain learns above this
CURVE_GAIN_INIT = 0.10  # fraction of the model's lateral accel taken off
CURVE_GAIN_MAX = 0.20
CURVE_KI = 0.04  # gain per m per s
MAX_CURVE_BIAS = 0.3  # m/s^2, hard cap on the curve term

LaneChangeState = log.LaneChangeState


def clip_offset_cm(offset_cm: int) -> int:
  return int(np.clip(offset_cm, -LANE_OFFSET_MAX_CM, LANE_OFFSET_MAX_CM))


def measure_lane(model_v2) -> tuple[float, float] | None:
  """(car position relative to lane center in metres, positive = right of center; lane width), None if lane lines are unreliable."""
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
  width = y_right - y_left
  if not LANE_WIDTH_RANGE[0] <= width <= LANE_WIDTH_RANGE[1]:
    return None
  return -(y_left + y_right) / 2.0, width


def measure_lane_position(model_v2) -> float | None:
  lane = measure_lane(model_v2)
  return lane[0] if lane is not None else None


class LaneOffset:
  def __init__(self):
    self.offset_cm = 0
    self.learned = 0.0  # m/s^2, kept across disengagements within a drive
    self.curve_gain = CURVE_GAIN_INIT  # kept across disengagements within a drive
    self.width_filter = FirstOrderFilter(0.0, WIDTH_FILTER_TC, DT_CTRL, initialized=False)
    self.filter = FirstOrderFilter(0.0, FILTER_TC, DT_CTRL)

  def update_param(self, offset_cm) -> None:
    self.offset_cm = clip_offset_cm(offset_cm or 0)

  def update(self, active: bool, v_ego: float, lane_change_state, model_v2=None,
             steering_pressed: bool = False, auto_center: bool = False, model_curvature: float = 0.0) -> float:
    """Returns the curvature bias to add to the model's desired curvature (openpilot convention, positive = right)."""
    if not active:
      self.filter.x = 0.0
      return 0.0

    target = 0.0
    curve_bias = 0.0
    if lane_change_state == LaneChangeState.off:
      model_lat_accel = model_curvature * v_ego ** 2
      if auto_center:
        self._learn(v_ego, model_v2, steering_pressed, model_lat_accel)
      target = 2.0 * (self.offset_cm / 100.0) / LOOKAHEAD_TIME ** 2
      if auto_center:
        target += self.learned
        curve_bias = float(np.clip(-self.curve_gain * model_lat_accel, -MAX_CURVE_BIAS, MAX_CURVE_BIAS))
      target = float(np.clip(target, -MAX_BIAS_LAT_ACCEL, MAX_BIAS_LAT_ACCEL))
      fade = float(np.interp(v_ego, FADE_SPEEDS, [0.0, 1.0]))
      target *= fade
      curve_bias *= fade

    # the curve term follows the model's request directly, the filter would make it late at curve entry
    lat_accel_bias = self.filter.update(target) + curve_bias
    return lat_accel_bias / max(v_ego, 1.0) ** 2

  def _learn(self, v_ego: float, model_v2, steering_pressed: bool, model_lat_accel: float = 0.0) -> None:
    if steering_pressed or v_ego < FADE_SPEEDS[1] or model_v2 is None:
      return
    lane = measure_lane(model_v2)
    if lane is None:
      return
    pos, width = lane
    width_avg = self.width_filter.update(width)
    if abs(width - width_avg) > MAX_WIDTH_CHANGE:
      return
    err = self.offset_cm / 100.0 - pos
    if abs(err) > MAX_ERR:
      return
    if abs(model_lat_accel) < STRAIGHT_LAT_ACCEL:
      self.learned = float(np.clip(self.learned + KI * err * DT_CTRL, -MAX_LEARNED, MAX_LEARNED))
    elif abs(model_lat_accel) > CURVE_LAT_ACCEL:
      # err > 0: the car should be further right. In a right curve (lat accel > 0) that means less outward bias.
      self.curve_gain = float(np.clip(self.curve_gain - CURVE_KI * err * np.sign(model_lat_accel) * DT_CTRL, 0.0, CURVE_GAIN_MAX))
