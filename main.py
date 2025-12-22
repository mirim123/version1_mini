import cv2
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
from collections import deque
from detect_stopped_car import detect_highway_stopped_vehicle
from get_speed_direction import detect_car_direction
from get_wrong_way_and_speeding import wrong_way_drive, get_real_speed
import time


file_path = f"./videos/20251222 10_00_02_[mon1].mp4"
cctv_id = 4
cap = cv2.VideoCapture(file_path)


# 프레임 속도를 1초를 환산
fps = cap.get(cv2.CAP_PROP_FPS)
# 속력을 계산할 때, 사용한 시간 7프레임 간격으로 속력 계산
one_second = 1 * 5 / fps


model = YOLO("best1.pt")
model1 = YOLO("yolov8n.pt")
# 실시간으로 6초 180개의 frame을 저장할 리스트 dq 설정
dq = deque(maxlen=180)
frame_num = 0
frame_count = 0

# 탐지된 차량 저장 딕션너리

vehicle_id = {}
recording_start = {}

# 빈 데이터 프레임 만들기
# columns = ["type", "dir", "speed", "datetime", "alarm", "file_name"]
# 예1) 1 : [버스, up, 160, 25-12-19_13:00:00, 과속, speeding_25-12-19_13:00:00.mp4 ]
# 예2) 1 : [자가용, down, 110, 25-12-19_13:00:00, 주정차, parking_25-12-19_13:00:00.mp4 ]

df = pd.DataFrame(
    columns=["type", "direction", "speed", "datetime", "illegal", "file_name"]
)


while cap.isOpened():
    frame_num += 1
    frame_count += 1
    direction = None

    if frame_count % 2 != 0:
        continue

    # print("Vehicle ID", vehicle_id)
    success, frame = cap.read()
    if not success:
        break

    # =================================================================
    # =======================  오토바이 & 사람 탐지  =====================
    # =================================================================

    # 사람과 오토바이 탐지 YOLO 모델 적용
    detect_motorcycle = model1.track(frame, verbose=False, persist=True, classes=[0, 3])

    for box in detect_motorcycle[0].boxes:
        if box.id is None:
            continue
        else:
            M_id = int(box.id)
            cls = int(box.cls)
            mx, my, _, _ = box.xywh[0].tolist()
            date_time = datetime.now()
            file_name = f"motorcycle_people_{date_time.strftime("%Y_%m_%d_%H_%M_%S")}"
            recording_start[M_id] = [(frame_num + 90) % 180, file_name, mx, my]
            # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]

            if cls == 0:
                print("경고! 고속도로 위에서 사람 발견")

            elif cls == 3:
                print("경고! 고속도로 위에서 오토바이 발견")

    # 분석하지 않는 화면 하얀색으로 전처리

    frame[:200, :] = 255  # 화면 상단 하얀색으로 전처리

    results = model.track(
        frame, verbose=False, conf=0.7, persist=True, classes=[0, 1, 2]
    )

    for box in results[0].boxes:

        if box.id is None:
            continue

        # 차량 ID와 그 차량의 중심값
        tid = int(box.id)

        cx, cy, _, _ = box.xywh[0].tolist()

        # =================================================================
        # ===============  차량 속도데이터 수집을 시작합니다. ===================
        # =================================================================

        # =================================================================
        # ========================  불법 주정차 탐지 =========================
        # =================================================================

        # 탐지 구역 지정:  이유: 탐지된 차량이 화면을 빠져나갈 때, 박스 라벨링 왜곡으로 오류 발생. 특히 트럭, 버스

        # 고속도로 위 정차되어 있는 차량 탐지.
        try:
            detect_stopped_car, stopped_id = detect_highway_stopped_vehicle(tid, cx, cy)

            if detect_stopped_car:
                print(f"차량 번호 : {stopped_id}: 고속도로 위에 주정차되어 있습니다.")

                # 현시간을 기준으로 전후 3초, 총 6초간 동영상 녹화
                stopped_car_datetime = datetime.now()
                # file_name 예시 parking_25-12-19_13:00:00.mp4
                file_name = f"parking[{tid}]_{stopped_car_datetime.strftime("%Y_%m_%d_%H_%M_%S")}"
                recording_start[tid] = [(frame_num + 90) % 180, file_name, cx, cy]
                # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]
                df.loc[tid, ["illegal", "file_name"]] = ["parking", f"{file_name}.mp4"]
        except FileNotFoundError as e:
            print(e, "파일을 못찾았어요")
        except Exception as e:
            print(e)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 탐지 존 영역 수정
        in_zone = 343 < cy < 356  # a, c, h           cctv  13차이 195   line = 350
        # in_zone = 300 < cy < 313  # b, e, f, g      cctv  13차이 195   line = 306
        # in_zone = 243 < c_y < 256  # d i             cctv  13차이 195   line = 250

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ==========================================================================
        # ==============================  차량 방향 및 속력 탐지 =============================
        # ==========================================================================
        if tid not in vehicle_id and in_zone:
            vehicle_id[tid] = 0
            # print(f"처음 Id: {id.item()}, y1의 좌표: {c_y}")

        result_dir_speed = detect_car_direction(tid, cx, cy, frame_num, one_second)

        if result_dir_speed is None:
            continue

        else:
            cls = int(box.cls)
            direction, speed_px = result_dir_speed
            data_time = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
            # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]
            cols = ["type", "direction", "speed", "datetime"]

            df.loc[tid, "type"] = cls
            df.loc[tid, "direction"] = direction
            df.loc[tid, "speed"] = speed_px
            df.loc[tid, "datetime"] = data_time

        # ==========================================================================
        # ==============================  역방향 탐지와 및 속력 보정 =============================
        # ==========================================================================

        # # 역방향 차량 탐지 및 속력 보정후 실제 속력
        # # 상행선 하행선의 차량 데이터 50개를 활용하여 최근접분류를 통해 역방향을 찾는다.
        if df.loc[tid, "speed"] is not None:
            car_direction = df.loc[tid, "direction"]
            speed_px1 = df.loc[tid, "speed"]
            cls = df.loc[tid, "type"]

            detect_wrong_way_car = wrong_way_drive(
                tid, cls, cx, cy, car_direction, speed_px1
            )

            # 차선에 따른 속력보정후 실제 속력
            real_speed = get_real_speed(cx)
            if real_speed is None:
                df.loc[tid, "speed"] = "processing"

            else:

                df.loc[tid, "speed"] = speed_px1 * real_speed

            # 아래의 car_direction은 차선과 관계없는 실제 차량의 주행방향
            # 최근접 분류 머신런닝을 통해 역주행 여부 판단
            if detect_wrong_way_car:
                print(f"경고!! 역주행 차량{tid}가 발견되었습니다.")

                wrong_way_datetime = datetime.now()
                # file_name 예시 parking_25-12-19_13:00:00.mp4
                file_name = f"wrong_way[{tid}]_{wrong_way_datetime.strftime("%Y_%m_%d_%H_%M_%S")}"
                recording_start[tid] = [(frame_num + 90) % 180, file_name, cx, cy]
                # column = ["type", "direction", "speed", "datetime", "illegal", "file_name"]
                df.loc[tid, ["illegal", "file_name"]] = [
                    "wrong_way",
                    f"{file_name}.mp4",
                ]

    # ==================================================================================
    # ====================================== 역주행 테스트 ===============================

    # if frame_num == 180:
    #     cls, cx, cy, car_direction, speed_px1
    #     wrong_way_drive(1, 100, 300, "up", 1000)
    #     df.loc["wrong_way"] = None
    #     df.loc["wrong_wya", ["illegal", "file_name"]] = ["wrong_way", "test"]
    #     print("wrong_way", df)
    # ==================================================================================

    # 리코딩 시작:  아이디 그리고 불법 종류 별로
    if recording_start:
        for key in list(recording_start.keys()):
            if recording_start[key][0] == frame_num:
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                file_out_path = f"results/{recording_start[key][1]}.mp4"
                out = cv2.VideoWriter(file_out_path, fourcc, fps, (720, 480))
                print("리코딩을 시작합니다")
                for f in dq:
                    out.write(f)

                out.release()
                recording_start.pop(key)  # 한 번만 실행

    if frame_num == 180:
        frame_num = 0

    result = results[0].plot()
    for key in recording_start.keys():
        cx = int(recording_start[key][2])
        cy = int(recording_start[key][3])
        cv2.circle(result, (cx, cy), 20, (0, 0, 255), 3)

    cv2.imshow("frame", result)

    # 녹화할 영상 dq에  담기
    dq.append(result)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        print(df.tail())

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
        break


# 데이터를 주기적으로 엑셀파일로 방출한다.
file_excel = f"./results/highway_traffic{cctv_id}_{int(time.time())}.xlsx"
df.to_excel(file_excel, index=False)


# DB로 저장한다.

cap.release()
cv2.destroyAllWindows()
