from types import SimpleNamespace

import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.lane_offset import (LaneOffset, LANE_OFFSET_MAX_CM, LOOKAHEAD_TIME, MAX_BIAS_LAT_ACCEL,
                                                          MAX_LEARNED, measure_lane_position)

LaneChangeState = log.LaneChangeState


def settle(lo: LaneOffset, v_ego: float, lane_change_state=LaneChangeState.off, steps: int = 1000) -> float:
  bias = 0.0
  for _ in range(steps):
    bias = lo.update(True, v_ego, lane_change_state)
  return bias


class TestLaneOffset:
  def test_zero_offset_is_noop(self):
    assert settle(LaneOffset(), 25.0) == 0.0

  def test_positive_offset_steers_right(self):
    # openpilot curvature is positive to the right
    lo = LaneOffset()
    lo.update_param(20)
    bias = settle(lo, 25.0)
    assert bias > 0
    assert abs(bias * 25.0 ** 2 - 2 * 0.20 / LOOKAHEAD_TIME ** 2) < 1e-3

    lo.update_param(-20)
    assert settle(lo, 25.0) < 0

  def test_param_is_clipped(self):
    lo = LaneOffset()
    lo.update_param(10_000)
    assert lo.offset_cm == LANE_OFFSET_MAX_CM
    lo.update_param(None)
    assert lo.offset_cm == 0

  def test_disabled_inactive_lane_change_and_low_speed(self):
    lo = LaneOffset()
    lo.update_param(30)
    settle(lo, 25.0)
    assert lo.update(False, 25.0, LaneChangeState.off) == 0.0
    assert lo.filter.x == 0.0

    assert abs(settle(lo, 25.0, LaneChangeState.laneChangeStarting)) < 1e-6
    assert abs(settle(lo, 2.0)) < 1e-6

  def test_ramps_in(self):
    lo = LaneOffset()
    lo.update_param(30)
    first = lo.update(True, 25.0, LaneChangeState.off)
    assert 0 < first < settle(lo, 25.0) * 0.05


# Personal: auto-center tests
def make_model(y_left=-1.8, y_right=1.8, prob=0.9, std=0.1, stds=True):
  def line(y):
    return SimpleNamespace(x=[0.0, 10.0, 50.0], y=[y, y, y])
  return SimpleNamespace(laneLines=[line(-5.0), line(y_left), line(y_right), line(5.0)],
                         laneLineProbs=[0.9, prob, prob, 0.9],
                         laneLineStds=[0.1, std, std, 0.1] if stds else [])


def make_model_at(pos, width=3.6, prob=0.9):
  # car at `pos` right of center: lane edges relative to the car
  return make_model(-width / 2 - pos, width / 2 - pos, prob)


def learn(lo, model, steps=1000, v_ego=25.0, **kw):
  for _ in range(steps):
    lo.update(True, v_ego, LaneChangeState.off, model, kw.get("steering_pressed", False), kw.get("auto_center", True))
  return lo.learned


class TestMeasureLanePosition:
  def test_centered(self):
    assert abs(measure_lane_position(make_model())) < 1e-9

  def test_sign_and_value(self):
    # car 0.3 m right of center: lines are further left of the car
    assert abs(measure_lane_position(make_model_at(0.3)) - 0.3) < 1e-9
    assert abs(measure_lane_position(make_model_at(-0.3)) + 0.3) < 1e-9

  def test_interpolates_at_zero(self):
    m = make_model()
    m.laneLines[1] = SimpleNamespace(x=[0.0, 10.0], y=[-1.5, -2.5])
    m.laneLines[2] = SimpleNamespace(x=[0.0, 10.0], y=[2.1, 2.1])
    assert abs(measure_lane_position(m) + 0.3) < 1e-9

  def test_invalid(self):
    assert measure_lane_position(make_model(prob=0.5)) is None
    assert measure_lane_position(make_model(std=0.3)) is None
    assert measure_lane_position(make_model(-1.0, 1.0)) is None  # too narrow
    assert measure_lane_position(make_model(-3.0, 3.0)) is None  # too wide
    assert measure_lane_position(make_model(std=0.3, stds=False)) is not None


class TestAutoCenterIntegrator:
  def test_learns_right_when_left_of_center(self):
    lo = LaneOffset()
    assert learn(lo, make_model_at(-0.3), 100) > 0
    lo = LaneOffset()
    assert learn(lo, make_model_at(0.3), 100) < 0

  def test_off_stays_zero(self):
    lo = LaneOffset()
    assert learn(lo, make_model_at(-0.3), 100, auto_center=False) == 0.0

  def test_offset_is_target(self):
    lo = LaneOffset()
    lo.update_param(30)
    assert learn(lo, make_model_at(0.3), 100) == pytest.approx(0.0, abs=1e-6)  # already at target

  def test_freezes(self):
    m = make_model_at(-0.3)
    base = LaneOffset()
    learn(base, m, 100)
    start = base.learned
    assert start > 0

    for kwargs in ({"steering_pressed": True}, {"v_ego": 8.0}):
      lo = LaneOffset()
      lo.learned = start
      learn(lo, m, 100, **kwargs)
      assert lo.learned == start

    lo = LaneOffset()
    lo.learned = start
    for _ in range(100):
      lo.update(True, 25.0, LaneChangeState.laneChangeStarting, m, False, True)
    assert lo.learned == start

    for bad in (make_model_at(-0.3, prob=0.3), make_model_at(-1.2)):  # unreliable, |err| > 1 m
      lo = LaneOffset()
      lo.learned = start
      learn(lo, bad, 100)
      assert lo.learned == start

    lo = LaneOffset()
    lo.learned = start
    for _ in range(100):
      lo.update(False, 25.0, LaneChangeState.off, m, False, True)
    assert lo.learned == start  # kept while inactive

  def test_freezes_while_lane_widens(self):
    # an exit lane forming on the right moves the measured center right; don't chase it
    lo = LaneOffset()
    learn(lo, make_model_at(0.0), 1000)
    assert lo.learned == 0.0
    for i in range(300):
      width = 3.6 + 0.6 * i / 300
      learn(lo, make_model_at(-(width - 3.6) / 2, width), 1)
    assert lo.learned < 0.01

  def test_clip(self):
    lo = LaneOffset()
    lo.learned = MAX_LEARNED - 1e-6
    learn(lo, make_model_at(-0.9), 1000)
    assert lo.learned == MAX_LEARNED
    lo.learned = -MAX_LEARNED + 1e-6
    learn(lo, make_model_at(0.9), 1000)
    assert lo.learned == -MAX_LEARNED

  def test_total_bias_capped(self):
    lo = LaneOffset()
    lo.update_param(50)  # ff = 0.25
    lo.learned = MAX_LEARNED
    bias = settle_model(lo, make_model_at(0.5), 25.0)
    assert abs(bias * 25.0 ** 2 - MAX_BIAS_LAT_ACCEL) < 1e-3


def settle_model(lo, model, v_ego, steps=1000):
  bias = 0.0
  for _ in range(steps):
    bias = lo.update(True, v_ego, LaneChangeState.off, model, False, True)
  return bias


def simulate(v, lookahead, offset_cm=0, bias=-0.3, duration=60.0, invalid_after=None):
  """Kinematic car in a 3.6 m lane (center y=0, +y right). The 'model' is pure pursuit toward a line `bias` m
  from center (negative = left), the lane lines are drawn from the true lane edges."""
  lo = LaneOffset()
  lo.update_param(offset_cm)
  y = bias  # starts on the model's line
  psi = 0.0
  max_pos = -9.0
  frozen_learned = None
  L = v * lookahead
  n = int(duration / DT_CTRL)
  for i in range(n):
    t = i * DT_CTRL
    invalid = invalid_after is not None and t >= invalid_after
    model = make_model_at(y, prob=0.3 if invalid else 0.9)
    if invalid and frozen_learned is None:
      frozen_learned = lo.learned
    kappa = 2.0 * (bias - (y + L * psi)) / L ** 2
    kappa += lo.update(True, v, LaneChangeState.off, model, False, True)
    psi += v * kappa * DT_CTRL
    y += v * psi * DT_CTRL
    max_pos = max(max_pos, y)
  return y, max_pos, lo.learned, frozen_learned


class TestAutoCenterClosedLoop:
  def test_centers_at_25(self):
    pos, max_pos, learned, _ = simulate(25.0, 2.0)
    assert abs(pos) < 0.05
    assert max_pos < 0.1
    assert learned > 0

  def test_offset_20cm(self):
    pos, _, _, _ = simulate(25.0, 2.0, offset_cm=20)
    assert abs(pos - 0.20) < 0.05

  def test_other_speeds_and_lookaheads(self):
    for v, la in ((12.0, 2.0), (25.0, 1.5), (25.0, 3.0), (12.0, 1.5), (12.0, 3.0)):
      pos, max_pos, _, _ = simulate(v, la)
      assert abs(pos) < 0.06, (v, la, pos)
      assert max_pos < 0.2, (v, la, max_pos)  # integrator overshoots a bit more with a long lookahead

  def test_frozen_when_lines_invalid(self):
    _, _, learned, frozen = simulate(25.0, 2.0, invalid_after=40.0)
    assert frozen is not None and frozen > 0
    assert learned == frozen
