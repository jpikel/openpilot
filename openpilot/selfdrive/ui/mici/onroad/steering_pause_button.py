import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Personal: top-center pill that pauses/resumes openpilot steering (stock ACC keeps working).
# selfdrived owns the state: the tap writes SteeringPaused and the pill shows selfdriveState.steeringPaused.
ZONE_WIDTH = 280
ZONE_HEIGHT = 76
PILL_PADDING = 22
PILL_HEIGHT = 50
PILL_TOP = 8
ICON_SIZE = 32
ICON_GAP = 8
FONT_SIZE = 30
PENDING_SECONDS = 1.0  # show the tapped state until selfdrived catches up


class SteeringPauseButton:
  def __init__(self):
    self._font = gui_app.font(FontWeight.BOLD)
    self._icon = gui_app.texture('icons_mici/wheel.png', ICON_SIZE, ICON_SIZE)
    self._pressed = False
    self._pending: bool | None = None
    self._pending_time = 0.0
    self._alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._rect = rl.Rectangle()

  def _zone(self) -> rl.Rectangle:
    r = self._rect
    return rl.Rectangle(r.x + (r.width - ZONE_WIDTH) / 2, r.y, ZONE_WIDTH, ZONE_HEIGHT)

  def _hit(self, pos: MousePos) -> bool:
    return self._alpha.x >= 0.5 and rl.check_collision_point_rec(pos, self._zone())

  def _paused(self) -> bool:
    actual = ui_state.sm['selfdriveState'].steeringPaused
    if self._pending is not None:
      if self._pending == actual or rl.get_time() - self._pending_time > PENDING_SECONDS:
        self._pending = None
      else:
        return self._pending
    return actual

  def handle_press(self, pos: MousePos) -> bool:
    self._pressed = self._hit(pos)
    return self._pressed

  def handle_release(self, pos: MousePos) -> bool:
    """Returns True if the tap was consumed by the button."""
    pressed, self._pressed = self._pressed, False
    hit = self._hit(pos)
    if not pressed:
      # never let a release on the button fall through to "go home"
      return hit
    if hit:
      self._pending = not self._paused()
      self._pending_time = rl.get_time()
      ui_state.params.put_bool("SteeringPaused", self._pending)
    return True

  def render(self, rect: rl.Rectangle, visible: bool, view_pressed: bool) -> None:
    self._rect = rect
    self._alpha.update(float(visible))
    if not visible or not view_pressed:
      self._pressed = False
    if self._alpha.x < 0.01:
      return

    a = self._alpha.x
    paused = self._paused()
    text = "steering off" if paused else "steering on"
    # fixed width (sized for the longer label) so the pill doesn't jump when toggled
    pill_width = ICON_SIZE + ICON_GAP + measure_text_cached(self._font, "steering off", FONT_SIZE).x + 2 * PILL_PADDING
    pill = rl.Rectangle(rect.x + (rect.width - pill_width) / 2, rect.y + PILL_TOP, pill_width, PILL_HEIGHT)
    if self._pressed:
      bg = rl.Color(255, 255, 255, int(110 * a))
    elif paused:
      bg = rl.Color(150, 30, 30, int(170 * a))
    else:
      bg = rl.Color(0, 0, 0, int(140 * a))
    rl.draw_rectangle_rounded(pill, 1.0, 16, bg)
    rl.draw_rectangle_rounded_lines_ex(pill, 1.0, 16, 2, rl.Color(255, 255, 255, int(120 * a)))

    size = measure_text_cached(self._font, text, FONT_SIZE)
    x = pill.x + (pill.width - (ICON_SIZE + ICON_GAP + size.x)) / 2
    fg = rl.Color(255, 255, 255, int((150 if paused else 235) * a))
    rl.draw_texture_ex(self._icon, rl.Vector2(x, pill.y + (pill.height - ICON_SIZE) / 2), 0.0, 1.0, fg)
    rl.draw_text_ex(self._font, text, rl.Vector2(x + ICON_SIZE + ICON_GAP, pill.y + (pill.height - size.y) / 2), FONT_SIZE, 0, fg)
