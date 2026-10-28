import itertools

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.selfdrive.selfdrived.always_on_lat import AlwaysOnLateral, ALT_EXP_ALWAYS_ON_LATERAL
from openpilot.selfdrive.selfdrived.events import Events, ET, EVENTS
from openpilot.selfdrive.selfdrived.state import StateMachine

EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState


def make_cs(available: bool, enabled: bool):
  CS = car.CarState.new_message()
  CS.cruiseState.available = available
  CS.cruiseState.enabled = enabled
  return CS


def make_car_events(*names):
  ret = []
  for n in names:
    e = log.OnroadEvent.new_message()
    e.name = n
    ret.append(e.as_reader())
  return ret


def names(car_events):
  return [e.name.raw for e in car_events]


class TestAlwaysOnLateral:
  def test_bit_matches_panda(self):
    assert ALT_EXP_ALWAYS_ON_LATERAL == 32

  def test_truth_table(self):
    for param, op_enabled, available, acc in itertools.product((False, True), repeat=4):
      aol = AlwaysOnLateral(param)
      CS = make_cs(available, acc)
      aol.update(CS, make_cs(available, acc), op_enabled)
      armed = param and op_enabled and available
      assert aol.armed == armed
      assert aol.steering_only(CS, op_enabled) == (armed and op_enabled and not acc)
      assert aol.must_disable(CS, op_enabled) == (param and op_enabled and not available)

  def test_disarm_when_main_off_or_op_disabled(self):
    aol = AlwaysOnLateral(True)
    CS = make_cs(True, True)
    assert aol.update(CS, CS, True)
    assert not aol.update(make_cs(False, False), CS, True)
    assert aol.update(CS, make_cs(False, False), True)
    assert not aol.update(CS, CS, False)

  def test_entered_steering_only_is_rising_edge(self):
    aol = AlwaysOnLateral(True)
    seq = [(True, True, False), (True, False, True), (True, False, False), (True, True, False), (True, False, True)]
    ret = []
    for available, acc, _ in seq:
      CS = make_cs(available, acc)
      aol.update(CS, CS, True)
      ret.append(aol.entered_steering_only)
    assert ret == [s[2] for s in seq]

  def test_filter_when_armed(self):
    aol = AlwaysOnLateral(True)
    CS = make_cs(True, False)
    aol.update(CS, CS, True)
    evs = make_car_events(EventName.pcmDisable, EventName.buttonCancel, EventName.wrongCruiseMode, EventName.steerOverride, EventName.accFaulted)
    assert names(aol.filter_car_events(evs, CS)) == [EventName.steerOverride, EventName.accFaulted]

  def test_filter_keeps_non_adaptive_when_acc_engaged(self):
    aol = AlwaysOnLateral(True)
    CS = make_cs(True, True)
    aol.update(CS, CS, True)
    evs = make_car_events(EventName.pcmDisable, EventName.buttonCancel, EventName.wrongCruiseMode)
    assert names(aol.filter_car_events(evs, CS)) == [EventName.wrongCruiseMode]

  def test_filter_noop_when_disarmed(self):
    evs = make_car_events(EventName.pcmDisable, EventName.buttonCancel, EventName.wrongCruiseMode)
    # param off
    aol = AlwaysOnLateral(False)
    CS = make_cs(True, False)
    aol.update(CS, CS, True)
    assert names(aol.filter_car_events(evs, CS)) == names(evs)
    # op not enabled
    aol = AlwaysOnLateral(True)
    aol.update(CS, CS, False)
    assert names(aol.filter_car_events(evs, CS)) == names(evs)

  def test_steering_only_alert_defined(self):
    assert ET.WARNING in EVENTS[EventName.steeringOnly]


class TestAlwaysOnLateralStateMachine:
  """Drive the filter + state machine the way selfdrived does, for a stock-ACC GM car."""

  def step(self, aol, sm, CS, brake=False, cancel=False):
    aol.update(CS, CS, sm.state != State.disabled)
    car_events = []
    if not CS.cruiseState.available:
      car_events.append(EventName.wrongCarMode)
    if not CS.cruiseState.enabled:
      car_events.append(EventName.pcmDisable)
    if cancel:
      car_events.append(EventName.buttonCancel)
    events = Events()
    events.add_from_msg(aol.filter_car_events(make_car_events(*car_events), CS))
    if aol.must_disable(CS, sm.state != State.disabled):
      events.add(EventName.pcmDisable)
    if brake and not aol.armed:
      events.add(EventName.pedalPressed)
    if CS.cruiseState.enabled and sm.state == State.disabled:
      events.add(EventName.pcmEnable)
    return sm.update(events)

  def test_brake_and_acc_cancel_keep_enabled_main_off_disables(self):
    # pcmEnable has no entry in this simplified harness, so force the engage through EVENTS having ET.ENABLE
    assert ET.ENABLE in EVENTS[EventName.pcmEnable]
    aol, sm = AlwaysOnLateral(True), StateMachine()
    assert self.step(aol, sm, make_cs(True, True))[0]
    # ACC cancelled by driver: still enabled
    assert self.step(aol, sm, make_cs(True, False), cancel=True)[0]
    # brake tap in steering-only: still enabled
    assert self.step(aol, sm, make_cs(True, False), brake=True)[0]
    # ACC re-engaged: still enabled
    assert self.step(aol, sm, make_cs(True, True))[0]
    # main off: disengage
    assert not self.step(aol, sm, make_cs(False, False))[0]
    # main off must not re-engage on an ACC edge
    assert not self.step(aol, sm, make_cs(False, True))[0]

  def test_param_off_disengages_on_brake_and_acc_off(self):
    aol, sm = AlwaysOnLateral(False), StateMachine()
    assert self.step(aol, sm, make_cs(True, True))[0]
    assert not self.step(aol, sm, make_cs(True, False))[0]
