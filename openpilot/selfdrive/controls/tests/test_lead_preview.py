from dataclasses import dataclass
import unittest

from openpilot.selfdrive.controls.lib.lead_preview import (
  DECEL_THRESHOLD,
  LeadPreview,
  LeadPreviewState,
  LeadPreviewSuppression,
  first_decel_crossing,
)


@dataclass
class Lead:
  present: bool = True
  modelProb: float = 0.95
  radarTrackId: int = 1
  dRel: float = 60.0
  yRel: float = 0.0
  vRel: float = -5.0
  vLeadK: float = 20.0
  aLeadK: float = 0.0


TIMES = [0.0, 2.0, 4.0, 6.0]
NO_DECEL = [0.0, 0.0, 0.0, 0.0]


def update(preview: LeadPreview, lead: Lead | None = None, *, accelerations=NO_DECEL, limiting_lead_index: int | None = None, **kwargs):
  inputs = {
    "engaged": True,
    "ego_speed": 30.0,
    "ego_acceleration": 0.0,
    "brake_pressed": False,
    "standstill": False,
    "lane_change_active": False,
    "fcw": False,
    "limiting_lead_index": limiting_lead_index,
    "plan_times": TIMES,
    "planned_accelerations": accelerations,
    "leads": (lead or Lead(), Lead(present=False)),
  }
  inputs.update(kwargs)
  return preview.update(**inputs)


def stabilize(preview: LeadPreview, lead: Lead | None = None):
  for _ in range(5):
    result = update(preview, lead)
  return result


class TestLeadPreview(unittest.TestCase):
  def test_first_decel_crossing_interpolates(self):
    self.assertAlmostEqual(first_decel_crossing(TIMES, [0.0, -0.2, -0.4, -0.5]), 3.0)
    self.assertEqual(first_decel_crossing(TIMES, [DECEL_THRESHOLD, -0.4, -0.5, -0.6]), 0.0)
    self.assertIsNone(first_decel_crossing(TIMES, NO_DECEL))

  def test_lead_must_be_stable_before_display(self):
    preview = LeadPreview(dt=0.1)
    for _ in range(4):
      self.assertEqual(update(preview).state, LeadPreviewState.HIDDEN)

    result = update(preview)
    self.assertEqual(result.state, LeadPreviewState.GREEN)
    self.assertTrue(result.track_stable)

  def test_preview_bands(self):
    for crossing_time, expected in ((2.0, LeadPreviewState.RED), (4.0, LeadPreviewState.AMBER)):
      with self.subTest(crossing_time=crossing_time):
        preview = LeadPreview(dt=0.1)
        stabilize(preview)
        accelerations = [0.0 if t < crossing_time else DECEL_THRESHOLD for t in TIMES]
        result = update(preview, accelerations=accelerations, limiting_lead_index=0)
        self.assertEqual(result.state, expected)
        self.assertTrue(result.decel_predicted)

  def test_non_lead_plan_stays_green(self):
    preview = LeadPreview(dt=0.1)
    stabilize(preview)
    result = update(preview, accelerations=[0.0, DECEL_THRESHOLD, -0.5, -0.5], limiting_lead_index=None)
    self.assertEqual(result.state, LeadPreviewState.GREEN)

  def test_prediction_outside_display_window_stays_green(self):
    preview = LeadPreview(dt=0.1)
    stabilize(preview)
    result = update(
      preview,
      accelerations=[0.0, 0.0, 0.0, DECEL_THRESHOLD],
      limiting_lead_index=0,
    )
    self.assertEqual(result.state, LeadPreviewState.GREEN)
    self.assertTrue(result.decel_predicted)
    self.assertEqual(result.predicted_decel_time, 6.0)

  def test_outer_timing_boundary_has_hysteresis(self):
    preview = LeadPreview(dt=0.1)
    stabilize(preview)
    result = update(
      preview,
      limiting_lead_index=0,
      plan_times=[0.0, 5.4, 7.0],
      accelerations=[0.0, DECEL_THRESHOLD, -0.5],
    )
    self.assertEqual(result.state, LeadPreviewState.AMBER)

    result = update(
      preview,
      limiting_lead_index=0,
      plan_times=[0.0, 5.7, 7.0],
      accelerations=[0.0, DECEL_THRESHOLD, -0.5],
    )
    self.assertEqual(result.state, LeadPreviewState.AMBER)

  def test_speed_matched_lead_returns_green(self):
    lead = Lead(vRel=-0.2)
    preview = LeadPreview(dt=0.1)
    stabilize(preview, lead)
    result = update(preview, lead, accelerations=[0.0, DECEL_THRESHOLD, -0.5, -0.5], limiting_lead_index=0)
    self.assertEqual(result.state, LeadPreviewState.GREEN)

  def test_suppression(self):
    cases = (
      ({"engaged": False}, LeadPreviewSuppression.DISENGAGED),
      ({"ego_speed": 5.0}, LeadPreviewSuppression.STOP_AND_GO),
      ({"brake_pressed": True}, LeadPreviewSuppression.DRIVER_BRAKING),
      ({"lane_change_active": True}, LeadPreviewSuppression.LANE_CHANGE),
      ({"fcw": True}, LeadPreviewSuppression.FCW),
    )
    for overrides, reason in cases:
      with self.subTest(reason=reason):
        preview = LeadPreview(dt=0.1)
        stabilize(preview)
        result = update(preview, **overrides)
        self.assertEqual(result.state, LeadPreviewState.HIDDEN)
        self.assertEqual(result.suppression, reason)

  def test_missing_and_low_confidence_leads_are_hidden(self):
    preview = LeadPreview(dt=0.1)
    result = update(preview, Lead(present=False))
    self.assertEqual(result.suppression, LeadPreviewSuppression.NO_LEAD)

    result = update(preview, Lead(modelProb=0.5))
    self.assertEqual(result.suppression, LeadPreviewSuppression.LOW_CONFIDENCE)

  def test_track_change_hides_speed_until_stable(self):
    preview = LeadPreview(dt=0.1)
    stabilize(preview)
    result = update(preview, Lead(radarTrackId=2, vLeadK=10.0))
    self.assertEqual(result.state, LeadPreviewState.HIDDEN)
    self.assertEqual(result.suppression, LeadPreviewSuppression.UNSTABLE_LEAD)
    self.assertEqual(result.lead_speed, 10.0)

  def test_actual_deceleration_onset_is_single_pulse_with_hysteresis(self):
    preview = LeadPreview(dt=0.1)
    stabilize(preview)
    self.assertTrue(update(preview, ego_acceleration=-0.31).actual_decel_onset)
    self.assertFalse(update(preview, ego_acceleration=-0.35).actual_decel_onset)
    self.assertFalse(update(preview, ego_acceleration=-0.25).actual_decel_onset)
    self.assertFalse(update(preview, ego_acceleration=-0.10).actual_decel_onset)
    self.assertTrue(update(preview, ego_acceleration=-0.31).actual_decel_onset)


if __name__ == "__main__":
  unittest.main()
