import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from hand_detection import get_hand_state
from rhythm_game import RhythmGame
from stage_loader import load_stage

# ==================================== TESTING = True, Asli = False
TEST_MOUSE_CONTROL = False
# =========================================================
GAME_PREVIEW_MARGIN_X = 220
GAME_PREVIEW_MARGIN_Y = 140
CAMERA_TARGET_FPS = 60


def clamp(value, low, high):
    return max(low, min(high, value))


def start_song(song_path):
    if pygame is None:
        print("Audio tidak bisa diputar: pygame belum terpasang")
        return False

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(song_path))
        pygame.mixer.music.set_volume(0.8)
        pygame.mixer.music.play()
        return True
    except pygame.error as error:
        print(f"Audio tidak bisa diputar: {error}")
        return False



def make_start_menu_canvas(width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    title = "so so!"
    subtitle = "Press SPACE or click to start"

    title_scale = 2.4
    title_thickness = 5
    subtitle_scale = 0.9
    subtitle_thickness = 2

    title_size, title_baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)
    subtitle_size, subtitle_baseline = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness)

    title_x = max(0, (width - title_size[0]) // 2)
    title_y = max(title_size[1] + 40, height // 2 - 30)
    subtitle_x = max(0, (width - subtitle_size[0]) // 2)
    subtitle_y = min(height - 40, title_y + 55)

    cv2.putText(
        canvas,
        title,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (subtitle_x, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        subtitle_scale,
        (180, 180, 180),
        subtitle_thickness,
        cv2.LINE_AA,
    )

    return canvas


def main():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    game = None
    stage = load_stage(Path(__file__).resolve().parent / "assets" / "stages" / "stage_01" / "stage.json")
    audio_started = None

    camera_width = None
    camera_height = None
    last_time = time.perf_counter()
    windows_ready = False
    mouse_position = {"x": None, "y": None, "active": False, "clicked": False}
    last_frame_time = time.time()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_position["x"] = x
            mouse_position["y"] = y
            mouse_position["active"] = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            mouse_position["clicked"] = True


    game_started = False
    last_gesture_state = "CLOSED" 

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if camera_width is None or camera_height is None:
            camera_height, camera_width = frame.shape[:2]

        if game is None:
            game = RhythmGame(
                camera_width,
                camera_height,
                chart_notes=stage.chart_notes,
                chart_offset_seconds=stage.chart_offset_seconds,
                preview_margin_x=GAME_PREVIEW_MARGIN_X,
                preview_margin_y=GAME_PREVIEW_MARGIN_Y,
            )


        if audio_started is None and game_started:
            # small delay to allow UI/audio initialization
            # time.sleep(1.0)
            audio_started = start_song(stage.song_path)
            game.reset_chart(
                chart_notes=stage.chart_notes,
                chart_offset_seconds=game.chart_offset_seconds,
            )

        if not windows_ready:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow(stage.window_title, cv2.WINDOW_NORMAL)
            cv2.namedWindow("Skin Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera", camera_width, camera_height)
            cv2.resizeWindow(stage.window_title, game.game_width, game.game_height)
            cv2.resizeWindow("Skin Mask", camera_width, camera_height)
            cv2.setMouseCallback(stage.window_title, on_mouse)
            windows_ready = True

        if TEST_MOUSE_CONTROL:
            annotated = frame.copy()
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hand_center = None
            gesture = "CLOSED"
        else:
            annotated, mask, hand_center, gesture = get_hand_state(frame)

        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        current_time = time.time()
        frame_delta = current_time - last_frame_time
        last_frame_time = current_time

        target_cursor = None
        instant_cursor = False
        if TEST_MOUSE_CONTROL:
            if mouse_position["active"]:
                target_cursor = game.cursor_from_window_position(
                    mouse_position["x"],
                    mouse_position["y"],
                )
                instant_cursor = True
        elif hand_center is not None and camera_width is not None and camera_height is not None:
            hand_x, hand_y = hand_center
            normalized_x = clamp(hand_x / max(1, camera_width), 0.0, 1.0)
            normalized_y = clamp(hand_y / max(1, camera_height), 0.0, 1.0)
            target_cursor = game.cursor_from_normalized(normalized_x, normalized_y)

        if game_started:
            if game.skill_cooldown_timer > 0.0:
                game.skill_cooldown_timer = max(0.0, game.skill_cooldown_timer - frame_delta)

            if game.is_skill_active:
                game.skill_duration_timer = max(0.0, game.skill_duration_timer - frame_delta)
                if game.skill_duration_timer <= 0.0:
                    game.is_skill_active = False
                    game.shield_arc_deg = game.normal_shield_arc

           
            if gesture == "OPEN" and last_gesture_state == "CLOSED":
                game.trigger_shield_skill()

            last_gesture_state = gesture

            game.update(dt, target_cursor, instant_cursor=instant_cursor)
            game_canvas = game.draw()
            
        else:
            game_canvas = make_start_menu_canvas(game.game_width, game.game_height)

        cv2.imshow("Camera", annotated)
        cv2.imshow(stage.window_title, game_canvas)
        cv2.imshow("Skin Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Start on spacebar or mouse click
        if not game_started and (key == ord(" ") or key == ord("s") or mouse_position.get("clicked")):
            game_started = True
            mouse_position["clicked"] = False

        # Restart on 'r' key
        if key == ord("r"):
            if pygame is not None and pygame.mixer.get_init():
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            audio_started = None
            game.reset_chart(
                chart_notes=stage.chart_notes,
                chart_offset_seconds=game.chart_offset_seconds,
            )
            game_started = True

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()