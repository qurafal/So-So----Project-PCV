import time
import math
import numpy as np
import cv2


def clamp(value, low, high):
    return max(low, min(high, value))


def lerp(current, target, alpha):
    return current + (target - current) * alpha


def rect_circle_collision(rect_x, rect_y, rect_w, rect_h, circle_x, circle_y, circle_radius):
    closest_x = clamp(circle_x, rect_x, rect_x + rect_w)
    closest_y = clamp(circle_y, rect_y, rect_y + rect_h)
    delta_x = circle_x - closest_x
    delta_y = circle_y - closest_y
    return (delta_x * delta_x + delta_y * delta_y) <= (circle_radius * circle_radius)


def angle_diff_deg(angle_a, angle_b):
    return (angle_a - angle_b + 180.0) % 360.0 - 180.0


def point_in_annulus_sector(point_x, point_y, center_x, center_y, inner_radius, outer_radius, angle_center_deg, angle_half_width_deg):
    dx = point_x - center_x
    dy = point_y - center_y
    distance = math.hypot(dx, dy)

    if distance < inner_radius or distance > outer_radius:
        return False

    point_angle = math.degrees(math.atan2(dy, dx))
    return abs(angle_diff_deg(point_angle, angle_center_deg)) <= angle_half_width_deg


def rect_intersects_annulus_sector(rect_x, rect_y, rect_w, rect_h, center_x, center_y, inner_radius, outer_radius, angle_center_deg, angle_half_width_deg):
    sample_points = (
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        (rect_x, rect_y + rect_h),
        (rect_x + rect_w * 0.5, rect_y + rect_h * 0.5),
        (rect_x + rect_w * 0.5, rect_y),
        (rect_x + rect_w, rect_y + rect_h * 0.5),
        (rect_x + rect_w * 0.5, rect_y + rect_h),
        (rect_x, rect_y + rect_h * 0.5),
    )

    for point_x, point_y in sample_points:
        if point_in_annulus_sector(
            point_x,
            point_y,
            center_x,
            center_y,
            inner_radius,
            outer_radius,
            angle_center_deg,
            angle_half_width_deg,
        ):
            return True

    return False


class Note:
    def __init__(self, side, game_width, game_height):
        self.side = side
        self.game_width = game_width
        self.game_height = game_height
        self.size = 46
        self.color = {
            "left": (255, 80, 80),
            "right": (80, 255, 80),
            "top": (80, 160, 255),
            "bottom": (255, 220, 80),
        }[side]

        self.center_x = game_width // 2
        self.center_y = game_height // 2
        self.hit = False

        speed = 200.0

        self.speed = speed
        self.spawn()

    def spawn(self):
        if self.side == "left":
            self.x = -self.size
            self.y = self.game_height // 2 - self.size // 2
            self.vx = self.speed
            self.vy = 0.0
        elif self.side == "right":
            self.x = self.game_width
            self.y = self.game_height // 2 - self.size // 2
            self.vx = -self.speed
            self.vy = 0.0
        elif self.side == "top":
            self.x = self.game_width // 2 - self.size // 2
            self.y = -self.size
            self.vx = 0.0
            self.vy = self.speed
        else:
            self.x = self.game_width // 2 - self.size // 2
            self.y = self.game_height
            self.vx = 0.0
            self.vy = -self.speed

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def has_reached_center(self, center_x, center_y, center_radius):
        return rect_circle_collision(
            self.x,
            self.y,
            self.size,
            self.size,
            center_x,
            center_y,
            center_radius,
        )

    def draw(self, canvas):
        x1 = int(self.x)
        y1 = int(self.y)
        x2 = int(self.x + self.size)
        y2 = int(self.y + self.size)
        x1 = clamp(x1, 0, self.game_width - 1)
        y1 = clamp(y1, 0, self.game_height - 1)
        x2 = clamp(x2, 0, self.game_width)
        y2 = clamp(y2, 0, self.game_height)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = self.color


class RhythmGame:
    def __init__(
        self,
        game_width=960,
        game_height=720,
        chart_notes=None,
        chart_offset_seconds=0.0,
        preview_margin_x=180,
        preview_margin_y=120,
    ):
        self.control_width = int(game_width)
        self.control_height = int(game_height)
        self.preview_margin_x = int(preview_margin_x)
        self.preview_margin_y = int(preview_margin_y)

        self.game_width = self.control_width + self.preview_margin_x * 2
        self.game_height = self.control_height + self.preview_margin_y * 2
        self.control_offset_x = self.preview_margin_x
        self.control_offset_y = self.preview_margin_y

        self.center_x = self.control_offset_x + self.control_width // 2
        self.center_y = self.control_offset_y + self.control_height // 2

        self.cursor_x = float(self.center_x)
        self.cursor_y = float(self.center_y)
        self.cursor_size = 12
        self.center_radius = 36
        # Shield settings: arc radius (from center), thickness, and angle width in degrees
        self.shield_radius = self.center_radius + 72
        self.shield_thickness = 20
        self.shield_arc_deg = 70
        self.notes = []
        self.score = 0
        self.misses = 0
        self.combo = 0
        self.chart_notes = list(chart_notes or [])
        self.chart_index = 0
        self.note_speed = 200
        self.chart_offset_seconds = float(chart_offset_seconds)
        self.start_time = time.perf_counter()

    def reset_chart(self, chart_notes=None, chart_offset_seconds=None):
        if chart_notes is not None:
            self.chart_notes = list(chart_notes)
        self.chart_index = 0

        self.score = 0
        self.misses = 0
        self.combo = 0

        if chart_offset_seconds is not None:
            self.chart_offset_seconds = float(chart_offset_seconds)

        self.start_time = time.perf_counter()

    def song_time(self):
        return time.perf_counter() - self.start_time - self.chart_offset_seconds

    def spawn_note(self, side):
        self.notes.append(Note(side, self.game_width, self.game_height))

    def clamp_cursor_to_control(self, cursor_x, cursor_y):
        min_x = self.cursor_size
        max_x = self.game_width - self.cursor_size
        min_y = self.cursor_size
        max_y = self.game_height - self.cursor_size
        return clamp(cursor_x, min_x, max_x), clamp(cursor_y, min_y, max_y)

    def cursor_from_window_position(self, mouse_x, mouse_y):
        return self.clamp_cursor_to_control(float(mouse_x), float(mouse_y))

    def cursor_from_normalized(self, normalized_x, normalized_y):
        cursor_x = normalized_x * self.game_width
        cursor_y = normalized_y * self.game_height
        return self.clamp_cursor_to_control(cursor_x, cursor_y)

    def travel_time_for_side(self, side, speed):
        note_size = 46
        if side in {"left", "right"}:
            distance = self.center_x + note_size
        else:
            distance = self.center_y + note_size
        return distance / speed

    def spawn_due_chart_notes(self):
        current_song_time = self.song_time()

        while self.chart_index < len(self.chart_notes):
            chart_note = self.chart_notes[self.chart_index]
            lead_time = self.travel_time_for_side(chart_note.side, self.note_speed)
            spawn_time = chart_note.time - lead_time

            if current_song_time < spawn_time:
                break

            self.spawn_note(chart_note.side)
            self.chart_index += 1

    def update(self, dt, target_cursor=None, instant_cursor=False):
        if target_cursor is not None:
            target_x, target_y = target_cursor
            target_x, target_y = self.clamp_cursor_to_control(target_x, target_y)
            if instant_cursor:
                self.cursor_x = target_x
                self.cursor_y = target_y
            else:
                self.cursor_x = lerp(self.cursor_x, target_x, 0.22)
                self.cursor_y = lerp(self.cursor_y, target_y, 0.22)

        self.spawn_due_chart_notes()

        remaining_notes = []
        for note in self.notes:
            note.update(dt)

            shield_angle = math.degrees(math.atan2(self.cursor_y - self.center_y, self.cursor_x - self.center_x))
            inner_radius = max(0.0, self.shield_radius - self.shield_thickness * 0.5)
            outer_radius = self.shield_radius + self.shield_thickness * 0.5

            if rect_intersects_annulus_sector(
                note.x,
                note.y,
                note.size,
                note.size,
                self.center_x,
                self.center_y,
                inner_radius,
                outer_radius,
                shield_angle,
                self.shield_arc_deg * 0.5,
            ):
                self.score += 1
                self.combo += 1
                continue

            if note.has_reached_center(self.center_x, self.center_y, self.center_radius):
                self.misses += 1
                self.combo = 0
                continue

            remaining_notes.append(note)

        self.notes = remaining_notes

    def draw(self):
        canvas = np.zeros((self.game_height, self.game_width, 3), dtype=np.uint8)

        center_x = self.center_x
        center_y = self.center_y

        x1 = self.control_offset_x
        y1 = self.control_offset_y
        x2 = self.control_offset_x + self.control_width
        y2 = self.control_offset_y + self.control_height

        canvas[y1:y2, x1:x2] = (12, 12, 12)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), 2)

        cv2.line(canvas, (center_x, 0), (center_x, self.game_height), (40, 40, 40), 1)
        cv2.line(canvas, (0, center_y), (self.game_width, center_y), (40, 40, 40), 1)
        # Draw shield arc facing the cursor

        dx_c = int(self.cursor_x - center_x)
        dy_c = int(self.cursor_y - center_y)
        ang = math.degrees(math.atan2(dy_c, dx_c))
        start_ang = ang - (self.shield_arc_deg / 2.0)
        end_ang = ang + (self.shield_arc_deg / 2.0)
        cv2.ellipse(
            canvas,
            (center_x, center_y),
            (int(self.shield_radius), int(self.shield_radius)),
            0,
            float(start_ang),
            float(end_ang),
            (160, 200, 255),
            int(self.shield_thickness),
        )

        cv2.circle(canvas, (center_x, center_y), self.center_radius, (220, 220, 220), 2)

        for note in self.notes:
            note.draw(canvas)

        cursor_x = int(self.cursor_x)
        cursor_y = int(self.cursor_y)
        cursor_size = self.cursor_size
        cv2.rectangle(
            canvas,
            (cursor_x - cursor_size, cursor_y - cursor_size),
            (cursor_x + cursor_size, cursor_y + cursor_size),
            (0, 0, 255),
            2,
        )
        cv2.circle(canvas, (cursor_x, cursor_y), 4, (0, 0, 255), -1)

        cv2.putText(canvas, f"Score: {self.score}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"Miss: {self.misses}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 180), 2)
        cv2.putText(canvas, f"Combo: {self.combo}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)
        cv2.putText(canvas, "Move hand to control cursor", (20, self.game_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return canvas
