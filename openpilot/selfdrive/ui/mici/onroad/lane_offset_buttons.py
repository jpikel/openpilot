import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.lane_offset import LANE_OFFSET_STEP_CM, clip_offset_cm
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Personal: big -/+ tap zones in the bottom corners of the onroad view to nudge lane position.
# The tap zone is larger than the drawn button so it is easy to hit while driving.
ZONE_SIZE = 150
BUTTON_SIZE = 96
BUTTON_MARGIN = 14
SIGN_LENGTH = 44
SIGN_THICKNESS = 10
VALUE_FONT_SIZE = 40
VALUE_FLASH_SECONDS = 2.5


def format_offset(offset_cm: int, is_metric: bool) -> str:
  if offset_cm == 0:
    return "centered"
  side = "right" if offset_cm > 0 else "left"
  if is_metric:
    return f"{abs(offset_cm)} cm {side}"
  return f"{round(abs(offset_cm) / 2.54)} in {side}"


class LaneOffsetButtons:
  def __init__(self):
    self._font = gui_app.font(FontWeight.BOLD)
    self._offset_cm = clip_offset_cm(ui_state.lane_offset_cm)
    self._pressed_direction = 0
    self._last_change_time = -VALUE_FLASH_SECONDS
    self._alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._rect = rl.Rectangle()

  def _zones(self) -> tuple[rl.Rectangle, rl.Rectangle]:
    r = self._rect
    left = rl.Rectangle(r.x, r.y + r.height - ZONE_SIZE, ZONE_SIZE, ZONE_SIZE)
    right = rl.Rectangle(r.x + r.width - ZONE_SIZE, r.y + r.height - ZONE_SIZE, ZONE_SIZE, ZONE_SIZE)
    return left, right

  def _hit(self, pos: MousePos) -> int:
    if self._alpha.x < 0.5:
      return 0
    left, right = self._zones()
    if rl.check_collision_point_rec(pos, left):
      return -1
    if rl.check_collision_point_rec(pos, right):
      return 1
    return 0

  def handle_press(self, pos: MousePos) -> bool:
    self._pressed_direction = self._hit(pos)
    return self._pressed_direction != 0

  def handle_release(self, pos: MousePos) -> bool:
    """Returns True if the tap was consumed by a button."""
    direction, self._pressed_direction = self._pressed_direction, 0
    hit = self._hit(pos)
    if direction == 0:
      # never let a release on a button fall through to "go home"
      return hit != 0
    if hit == direction:
      self._offset_cm = clip_offset_cm(self._offset_cm + direction * LANE_OFFSET_STEP_CM)
      ui_state.params.put("LaneOffsetCm", self._offset_cm)
      self._last_change_time = rl.get_time()
    # swallow the release either way so a slipped finger doesn't leave the onroad view
    return True

  def render(self, rect: rl.Rectangle, visible: bool, view_pressed: bool) -> None:
    self._rect = rect
    # pick up changes made elsewhere (settings reset), but don't fight a tap whose write may not have landed yet
    if rl.get_time() - self._last_change_time > VALUE_FLASH_SECONDS + 2.0:
      self._offset_cm = clip_offset_cm(ui_state.lane_offset_cm)

    self._alpha.update(float(visible))
    # a press that became a swipe or left the view never gets a release
    if not visible or not view_pressed:
      self._pressed_direction = 0
    if self._alpha.x < 0.01:
      return

    left, right = self._zones()
    for direction, zone in ((-1, left), (1, right)):
      cx = zone.x + BUTTON_MARGIN + BUTTON_SIZE / 2 if direction < 0 else zone.x + zone.width - BUTTON_MARGIN - BUTTON_SIZE / 2
      cy = zone.y + zone.height - BUTTON_MARGIN - BUTTON_SIZE / 2
      pressed = self._pressed_direction == direction
      bg = rl.Color(255, 255, 255, int(110 * self._alpha.x)) if pressed else rl.Color(0, 0, 0, int(140 * self._alpha.x))
      rl.draw_circle(int(cx), int(cy), BUTTON_SIZE / 2, bg)
      rl.draw_circle_lines(int(cx), int(cy), BUTTON_SIZE / 2, rl.Color(255, 255, 255, int(120 * self._alpha.x)))

      fg = rl.Color(255, 255, 255, int(235 * self._alpha.x))
      rl.draw_rectangle_rec(rl.Rectangle(cx - SIGN_LENGTH / 2, cy - SIGN_THICKNESS / 2, SIGN_LENGTH, SIGN_THICKNESS), fg)
      if direction > 0:
        rl.draw_rectangle_rec(rl.Rectangle(cx - SIGN_THICKNESS / 2, cy - SIGN_LENGTH / 2, SIGN_THICKNESS, SIGN_LENGTH), fg)

    # current offset: bright for a moment after a tap, then dim (hidden when centered)
    flashing = rl.get_time() - self._last_change_time < VALUE_FLASH_SECONDS
    if flashing or self._offset_cm != 0:
      text = format_offset(self._offset_cm, ui_state.is_metric)
      size = measure_text_cached(self._font, text, VALUE_FONT_SIZE)
      pos = rl.Vector2(rect.x + (rect.width - size.x) / 2, rect.y + rect.height - size.y - BUTTON_MARGIN)
      text_alpha = (1.0 if flashing else 0.55) * self._alpha.x
      rl.draw_text_ex(self._font, text, rl.Vector2(pos.x + 2, pos.y + 2), VALUE_FONT_SIZE, 0, rl.Color(0, 0, 0, int(160 * text_alpha)))
      rl.draw_text_ex(self._font, text, pos, VALUE_FONT_SIZE, 0, rl.Color(255, 255, 255, int(255 * text_alpha)))
