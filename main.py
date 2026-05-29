import cv2
import numpy as np


def bgr_to_ycrcb(frame):
    frame = frame.astype(np.float32)
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]

    y = 0.114 * b + 0.587 * g + 0.299 * r
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0

    return y, cr, cb


def skin_mask(frame):
    y, cr, cb = bgr_to_ycrcb(frame)

    mask = (
        (y > 70.0)
        & (cr >= 133.0)
        & (cr <= 173.0)
        & (cb >= 77.0)
        & (cb <= 127.0)
    )

    return mask


def find_largest_component(mask, min_area=3000):
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


def detect_hand(frame):
    mirrored = frame[:, ::-1].copy()
    mask = skin_mask(mirrored)

    hand_box = find_largest_component(mask)
    annotated = mirrored.copy()

    if hand_box is not None:
        x, y, w, h, area = hand_box
        draw_rectangle(annotated, x, y, w, h, color=(0, 255, 0), thickness=3)

        highlighted = annotated[y:y + h, x:x + w]
        if highlighted.size > 0:
            overlay = highlighted.astype(np.float32)
            overlay[:, :, 1] = np.clip(overlay[:, :, 1] + 35.0, 0, 255)
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 20.0, 0, 255)
            annotated[y:y + h, x:x + w] = overlay.astype(np.uint8)

    return mirrored, annotated, mask.astype(np.uint8) * 255


def main():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        original, annotated, mask = detect_hand(frame)

        cv2.imshow("Camera", original)
        cv2.imshow("Hand Detection", annotated)
        cv2.imshow("Skin Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()