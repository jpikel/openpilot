from openpilot.cereal import log

EventName = log.OnroadEvent.EventName

# Personal: always-on lateral. Must match ALT_EXP_ALWAYS_ON_LATERAL in opendbc/safety/__init__.py (panda).
# Panda, with this bit on a GM camera car with stock cruise: sets controls_allowed on a stock ACC rising edge
# (only while CruiseMainOn), does NOT clear it on brake/regen/ACC off, and DOES clear it when CruiseMainOn goes off.
ALT_EXP_ALWAYS_ON_LATERAL = 32

# Car events that normally disengage on brake/ACC cancel. Dropped while armed.
ACC_DISENGAGE_EVENTS = (EventName.pcmDisable, EventName.buttonCancel)


class AlwaysOnLateral:
  """
  Tracks whether openpilot may keep steering without stock ACC ("steering-only").
  Armed while openpilot is enabled and the cruise main switch is on; disarmed otherwise.
  Mirrors panda's behavior so openpilot and panda agree on controls_allowed.
  """
  def __init__(self, enabled_by_param: bool):
    self.enabled_by_param = enabled_by_param
    self.armed = False
    self.steering_only_prev = False
    self.entered_steering_only = False  # rising edge of steering_only, set by update()

  def update(self, CS, CS_prev, op_enabled: bool) -> bool:
    self.armed = self.enabled_by_param and op_enabled and CS.cruiseState.available
    steering_only = self.steering_only(CS, op_enabled)
    self.entered_steering_only = steering_only and not self.steering_only_prev
    self.steering_only_prev = steering_only
    return self.armed

  def steering_only(self, CS, op_enabled: bool) -> bool:
    return self.armed and op_enabled and not CS.cruiseState.enabled

  def must_disable(self, CS, op_enabled: bool) -> bool:
    # Main switch off: panda clears controls_allowed, so openpilot has to disengage too
    return self.enabled_by_param and op_enabled and not CS.cruiseState.available

  def filter_car_events(self, car_events, CS):
    """Drop events that would disengage on brake / stock ACC cancel while armed."""
    if not self.armed:
      return list(car_events)
    drop = ACC_DISENGAGE_EVENTS
    # Off-state ACC reports as non-adaptive on some GM cars; only a real (engaged) non-adaptive cruise should disengage
    if not CS.cruiseState.enabled:
      drop = (*drop, EventName.wrongCruiseMode)
    return [e for e in car_events if e.name.raw not in drop]
