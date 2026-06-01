import numpy as np
import cv2


SKIN_H_MIN = 0.0
SKIN_H_MAX = 20.0
SKIN_S_MIN = 20.0
SKIN_S_MAX = 150.0
SKIN_V_MIN = 100.0

BOX_SHRINK_RATIO = 0.18
HAND_CENTER_SQUARE_SIZE = 24
HAND_HIGHLIGHT_SATURATION_BOOST = 35.0
HAND_HIGHLIGHT_VALUE_BOOST = 20.0
MIN_COMPONENT_AREA = 3000
HAND_DETECT_SCALE = 0.5


def bgr_to_hsv(frame):
    frame = frame.astype(np.float32)
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    saturation = np.zeros_like(cmax)
    value = cmax / 255.0

    epsilon = 1e-6
    non_zero = delta > epsilon

    red_max = (cmax == r) & non_zero
    hue[red_max] = 60.0 * (((g[red_max] - b[red_max]) / (delta[red_max] + epsilon)) % 6.0)

    green_max = (cmax == g) & non_zero
    hue[green_max] = 60.0 * (((b[green_max] - r[green_max]) / (delta[green_max] + epsilon)) + 2.0)

    blue_max = (cmax == b) & non_zero
    hue[blue_max] = 60.0 * (((r[blue_max] - g[blue_max]) / (delta[blue_max] + epsilon)) + 4.0)

    saturation[cmax > epsilon] = delta[cmax > epsilon] / (cmax[cmax > epsilon] + epsilon)

    h = hue / 2.0
    s = saturation * 255.0
    v = value * 255.0

    return h, s, v


def skin_mask(frame):
    h, s, v = bgr_to_hsv(frame)
    return (
        (h >= SKIN_H_MIN)
        & (h <= SKIN_H_MAX)
        & (s >= SKIN_S_MIN)
        & (s <= SKIN_S_MAX)
        & (v >= SKIN_V_MIN)
    )


def find_largest_component(mask, min_area=MIN_COMPONENT_AREA):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component = None
    best_area = 0

    candidate_y, candidate_x = np.where(mask)

    for start_y, start_x in zip(candidate_y, candidate_x):
        if visited[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        area = 0
        min_x = max_x = start_x
        min_y = max_y = start_y

        while stack:
            current_y, current_x = stack.pop()
            area += 1

            if current_x < min_x:
                min_x = current_x
            if current_x > max_x:
                max_x = current_x
            if current_y < min_y:
                min_y = current_y
            if current_y > max_y:
                max_y = current_y

            y0 = max(0, current_y - 1)
            y1 = min(height, current_y + 2)
            x0 = max(0, current_x - 1)
            x1 = min(width, current_x + 2)

            for neighbor_y in range(y0, y1):
                for neighbor_x in range(x0, x1):
                    if not visited[neighbor_y, neighbor_x] and mask[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))

        if area > best_area:
            best_area = area
            best_component = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area)

    if best_component is None or best_area < min_area:
        return None

    return best_component


def draw_rectangle(frame, x, y, w, h, color=(0, 255, 0), thickness=2):
    height, width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    for offset in range(thickness):
        top = y1 + offset
        bottom = y2 - 1 - offset
        left = x1 + offset
        right = x2 - 1 - offset

        if top < y2:
            frame[top, x1:x2] = color
        if bottom >= y1:
            frame[bottom, x1:x2] = color
        if left < x2:
            frame[y1:y2, left] = color
        if right >= x1:
            frame[y1:y2, right] = color


def draw_filled_square(frame, center_x, center_y, size, color=(0, 0, 255)):
    height, width = frame.shape[:2]
    half_size = max(1, size // 2)

    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(width, center_x + half_size + 1)
    y2 = min(height, center_y + half_size + 1)

    frame[y1:y2, x1:x2] = color


def get_hand_state(frame):
    mirrored = frame[:, ::-1].copy()

    scale = HAND_DETECT_SCALE
    small = cv2.resize(mirrored, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    small_mask = skin_mask(small)

    scaled_min_area = max(50, int(MIN_COMPONENT_AREA * (scale * scale)))
    hand_box_small = find_largest_component(small_mask, min_area=scaled_min_area)

    annotated = mirrored.copy()
    hand_center = None

    if hand_box_small is not None:
        sx, sy, sw, sh, _area = hand_box_small
        shrink_x = max(1, int(sw * BOX_SHRINK_RATIO))
        shrink_y = max(1, int(sh * BOX_SHRINK_RATIO))

        sx = sx + shrink_x
        sy = sy + shrink_y
        sw = max(1, sw - (2 * shrink_x))
        sh = max(1, sh - (2 * shrink_y))

        x = int(sx / scale)
        y = int(sy / scale)
        w = int(sw / scale)
        h = int(sh / scale)

        draw_rectangle(annotated, x, y, w, h, color=(0, 255, 0), thickness=3)

        center_x = x + w // 2
        center_y = y + h // 2
        hand_center = (center_x, center_y)
        draw_filled_square(annotated, center_x, center_y, HAND_CENTER_SQUARE_SIZE, color=(0, 0, 255))

        highlighted = annotated[y:y + h, x:x + w]
        if highlighted.size > 0:
            overlay = highlighted.astype(np.float32)
            overlay[:, :, 1] = np.clip(overlay[:, :, 1] + HAND_HIGHLIGHT_SATURATION_BOOST, 0, 255)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] + HAND_HIGHLIGHT_VALUE_BOOST, 0, 255)
            annotated[y:y + h, x:x + w] = overlay.astype(np.uint8)

    mask_display = (small_mask.astype(np.uint8) * 255)
    mask_up = cv2.resize(mask_display, (mirrored.shape[1], mirrored.shape[0]), interpolation=cv2.INTER_NEAREST)

    return annotated, mask_up, hand_center
