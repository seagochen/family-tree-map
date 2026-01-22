import json
import os
import time
import traceback
from argparse import Namespace

import cv2
import cv2 as cv
from cv2 import CAP_PROP_FPS, VideoCapture
from numpy import ndarray

from scripts.common.db_proxy import DBProxy
from scripts.demo.predictor import Predictor

from collections import deque
from itertools import islice


class App:
    def run(self, args: Namespace) -> bool:
        result = True
        try:
            db_proxy = DBProxy()
            with db_proxy.connection(
                    args.db_host, args.db_name, args.db_user, args.db_secret
            ):
                frame_interval_ms = db_proxy.get_config_as_int("frame_sampling_interval")
                frame_cnt = db_proxy.get_config_as_int("clip_frame_count")

            pred = Predictor(
                args.spatial_model,
                args.spatial_detection_threshold,
                args.temporal_model,
            )

            stream = VideoCapture(args.stream)
            if not stream.isOpened():
                raise IOError(f"Failed to open stream: {args.stream}")

            # 获取视频宽高
            width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = stream.get(cv.CAP_PROP_FPS)

            # 时间戳命名
            output_filename = time.strftime("output_video_%Y%m%d_%H%M%S.mp4")
            print(f"Saving detection video to {output_filename}")

            # 初始化 VideoWriter
            output = cv.VideoWriter(output_filename,
                                    cv.VideoWriter_fourcc(*'mp4v'),
                                    fps, (width, height))

            self.detect(pred, stream, 1e-3 * frame_interval_ms, frame_cnt, output)
        except KeyboardInterrupt:
            print("Interrupted by user")
        except BaseException:
            result = False
            print(f"Error: {traceback.format_exc()}")
        finally:
            if 'output' in locals():
                output.release()
            if 'stream' in locals():
                stream.release()
            cv.destroyAllWindows()
            print("Resources released.")
        return result

    def detect(
            self,
            predictor: Predictor,
            stream: VideoCapture,
            sampling_interval_sec: float,
            clip_frame_count: int,
            output: cv.VideoWriter
    ):
        frame_interval_sec = 1 / stream.get(CAP_PROP_FPS)
        frame_buf: list[tuple[ndarray, float]] = []
        now = time.perf_counter()
        next_read_tm = now
        next_sampling_tm = now
        stream_start_tm = now

        # 记录每个clip的fire检测结果（True/False）
        event_state_buf = deque(maxlen=100)
        clip_times = deque(maxlen=100)

        ENTER_THRESHOLD = 4  # 至少 4 个 clip 是火灾才进入状态
        EXIT_COOLDOWN = 2  # 连续 2 个 clip 判定为非火灾才退出状态

        event_active = False
        fire_start_time = None
        event_id = 0
        fire_events = []  # 所有事件的记录
        output_dir = 'events'
        os.makedirs(output_dir, exist_ok=True)
        frame_index = 0

        while True:
            now = time.perf_counter()
            if now < next_read_tm:
                time.sleep(frame_interval_sec / 7)
                continue

            # Read a frame
            succeeded, frame = stream.read()
            if not succeeded:
                print("Video stream ended. Releasing resources...")
                stream.release()
                cv.destroyAllWindows()
                break
            assert (frame is not None)

            next_read_tm = now + frame_interval_sec
            if now < next_sampling_tm:
                continue

            # Sample the frame
            timestamp_sec = frame_index * sampling_interval_sec
            frame_buf.append((frame, timestamp_sec))
            frame_index += 1
            next_sampling_tm = now + sampling_interval_sec

            # Wait all frames of a clip before feeding them to the model, since
            # there may be no enough frames to form a complete clip (e.g., a PTZ
            # camera turns to next preset before enough frames are collected).
            if len(frame_buf) < clip_frame_count:
                continue

            pred = predictor.predict(tuple(f for f, _ in frame_buf))
            mid_idx = clip_frame_count // 2
            clip_times.append(frame_buf[mid_idx][1])

            # Make a prediction
            #if pred := predictor.predict(tuple(f for f, _ in frame_buf)):
            if pred and pred.bboxes and any(pred.bboxes):
                event_state_buf.append(True)
                det_time = frame_buf[pred.frame_index][1] - stream_start_tm
                bboxes = '\n'.join(' ' * 4 + str(b) for b in pred.bboxes)
                print(
                    f"Fire detected: time({det_time}), score({pred.score}),\n "
                    f"bboxes:\n{bboxes},\n "
                    f"spatial_confidences:\n{pred.spatial_confidences},\n"
                    f"typess:\n{pred.types}"
                )
            else:
                event_state_buf.append(False)
                print("Nothing detected")

            # 遍历滑动窗口中的每一帧
            for frame_idx, (img, timestamp) in enumerate(frame_buf[:(clip_frame_count // 2)]):
                isPrint = 1
                if pred and pred.bboxes and any(pred.bboxes):
                    # 对应这一帧的所有标注框
                    bbox_list = pred.bboxes[frame_idx]
                    confidence_list = pred.spatial_confidences[frame_idx]
                    types = pred.types[frame_idx]
                    # 遍历所有标注框，绘制到图片上
                    for b, conf, type in zip(bbox_list, confidence_list, types):
                        print(
                            f"print b ({b}), conf ({conf}),type ({type})\n "
                        )
                        if type == 2:
                            isPrint = 0
                            continue
                        x_min, y_min, bw, bh = b
                        h, w = img.shape[:2]
                        x1 = round(x_min * w)
                        y1 = round(y_min * h)
                        x2 = round((x_min + bw) * w)
                        y2 = round((y_min + bh) * h)

                        if type == 0:
                            type_label = "smoke"
                            box_color = (255, 0, 0)  # 蓝色
                        elif type == 1:
                            type_label = "fire"
                            box_color = (0, 0, 255)  # 红色
                        else:
                            type_label = "none"
                            box_color = (0, 255, 0)  # 绿色

                        # 画框
                        img = cv.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                        # 绘制置信度
                        confidence_text = f"Conf: {conf:.2f}"
                        cv2.putText(img, confidence_text, (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                        cv2.putText(img, type_label, (x1, y2 + 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                if isPrint == 1:
                    # 实时显示窗口
                    cv.imshow("Real-Time Fire Detection", img)
                    output.write(img)
                    cv.waitKey(1)

            # Majority Voting 判断事件状态
            fire_votes = sum(event_state_buf)
            if not event_active:
                majority_fire = fire_votes >= ENTER_THRESHOLD
            else:
                length = len(event_state_buf)
                majority_fire = not all(s == False for s in islice(event_state_buf, max(0, length - EXIT_COOLDOWN), length))

            # 进入火灾状态
            if not event_active and majority_fire:
                event_active = True
                event_id += 1
                # 修正火灾开始时间为连续火clip窗口的第一个clip的时间
                fire_start_time = None
                for i in range(len(event_state_buf) - ENTER_THRESHOLD + 1):
                    if all(event_state_buf[i + j] for j in range(ENTER_THRESHOLD)):
                        fire_start_time = clip_times[i]
                        break
                if fire_start_time is None:
                    fire_start_time = clip_times[0]  # fallback
                print(f"[火災開始]　開始時間: {fire_start_time:.2f} 秒")

            # 离开火灾状态
            if event_active and not majority_fire:
                event_active = False
                # 找最后一个为 True 的 clip 时间
                for i in range(len(event_state_buf) - 1, -1, -1):
                    if event_state_buf[i]:
                        fire_end_time = clip_times[i]
                        break
                    else:
                        fire_end_time = clip_times[0]  # fallback

                duration = fire_end_time - fire_start_time
                print(f"[火災終了]　終了時間: {fire_end_time:.2f} 秒，時速時間: {duration:.2f} 秒")

                # 记录事件
                fire_events.append({
                    "event_id": f"event_{event_id:03d}",
                    "start_time": round(fire_start_time, 2),
                    "end_time": round(fire_end_time, 2),
                    "duration": round(duration, 2),
                })
                print(f"[事件 {event_id:03d}] event_state_buf = {event_state_buf}")
                print(f"[事件 {event_id:03d}] clip_times = {clip_times}")

                # 离开火灾状态后清空历史
                event_state_buf.clear()
                clip_times.clear()

            # 状态维持中
            if event_active:
                print("🔥 火灾事件进行中")
            else:
                print("✅ 安全状态")

            # Slide window
            del frame_buf[:clip_frame_count // 2]

        # 视频结束时，如火灾仍在进行中，强制关闭事件并写入
        if event_active:
            # 找最后一个为 True 的 clip 时间
            for i in range(len(event_state_buf) - 1, -1, -1):
                if event_state_buf[i]:
                    #fire_end_time = clip_times[i] + sampling_interval_sec * (clip_frame_count - 1)
                    fire_end_time = clip_times[-1][1]
                    break
            else:
                fire_end_time = clip_times[-1][1]  # fallback（不应常触发）

            duration = fire_end_time - fire_start_time
            print(f"[强制结束] 视频结束仍处于火灾状态，终止时间: {fire_end_time:.2f} 秒，持续时间: {duration:.2f} 秒")

            fire_events.append({
                "event_id": f"event_{event_id:03d}",
                "start_time": round(fire_start_time, 2),
                "end_time": round(fire_end_time, 2),
                "duration": round(duration, 2),
            })

            print(f"[事件 {event_id:03d}] event_state_buf = {event_state_buf}")

            # 离开火灾状态后清空历史
            event_state_buf.clear()
            clip_times.clear()

            # if video_writer:
            #     video_writer.release()
            #     video_writer = None

        # 所有事件导出
        if fire_events:
            json_path = os.path.join(output_dir, "events.json")
            print(f"[事件状态] clip_times: {clip_times}")
            print(f"[事件状态] event_state_buf: {event_state_buf}")

            from datetime import timedelta
            def format_seconds(seconds: float) -> str:
                td = timedelta(seconds=seconds)
                total_seconds = td.total_seconds()
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                secs = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

            # 转换 start_time 和 end_time 为 HH:MM:SS.sss 字符串
            for ev in fire_events:
                ev["start_time_hms"] = format_seconds(ev["start_time"])
                ev["end_time_hms"] = format_seconds(ev["end_time"])

            with open(json_path, 'w') as jf:
                json.dump({"events": fire_events}, jf, indent=2)

            print(f"📄 火灾事件已保存: {json_path}")

        print("Detection complete. Video saved.")
        stream.release()
        cv2.destroyAllWindows()
