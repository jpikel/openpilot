from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.lane_offset import LaneOffset, LANE_OFFSET_MAX_CM, LOOKAHEAD_TIME

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
