import math


vehicle_id_direction = set()

# 임시 저장 변수 지정
pre_location = {}


def detect_car_direction(
    tid, cx, cy, frame_num, one_second
) -> tuple[str, float] | None:

    if tid not in vehicle_id_direction:
        pre_location[tid] = [(frame_num + 5) % 180, cx, cy]
        vehicle_id_direction.add(tid)
        return None

    if pre_location[tid] is None:
        return None

    prev_frame, prev_cx, prev_cy = (
        pre_location[tid][0],
        pre_location[tid][1],
        pre_location[tid][2],
    )

    # 5프레임 약 0.5초후에 속력 파악 : (pre_frame = frame_num + 5)

    if frame_num == prev_frame:
        # 차량 이동 계산
        dx = cx - prev_cx
        dy = cy - prev_cy
        move_dist = math.hypot(dx, dy)
        speed = move_dist / one_second

        # 방향 판단 (OpenCV 좌표계 기준)
        if cy < prev_cy:
            direction = "up"

        else:
            direction = "down"

        pre_location[tid] = None
        return direction, speed

    else:
        return None
