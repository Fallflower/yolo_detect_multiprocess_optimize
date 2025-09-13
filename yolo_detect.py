import multiprocessing
import time
import cv2
import torch
from ultralytics import YOLO
from collections import OrderedDict


def set_device():
    """
    设置使用的设备，优先使用CUDA（GPU）
    """
    _device = 'cpu'
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        _device = 'cuda'
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"GPU accelerate: {device_name} (device {current_device}/{gpu_count - 1})")
    else:
        print("Warning: CUDA is unavailable, turn to CPU instead.")

    return _device


model_path = 'model/yolo11m.pt'
threshold = 0.5


def read_frame(input_path, frame_queue, max_size=16, num_processes=4):
    cap = cv2.VideoCapture(input_path)
    frame_index = 0
    while True:
        if frame_queue.qsize() < max_size:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame_index, frame))
            frame_index += 1
        else:
            time.sleep(0.01)
    cap.release()
    for _ in range(num_processes):
        frame_queue.put((None, None))
    print('读取进程结束')


def detect(frame_queue, result_queue, device, vpid):
    model = YOLO(model_path)
    model.to(device)
    while True:
        frame_data = frame_queue.get()
        if frame_data[0] is None:  # 收到结束信号
            break
        frame_index, frame = frame_data
        results = model(frame, conf=threshold, device=device, verbose=False)

        annotated_frame = results[0].plot()

        result_queue.put((frame_index, annotated_frame))
    result_queue.put((None, vpid))
    print("检测进程 [%i] 结束" % vpid)


def write(fps, width, height, total_frames, output_path, result_queue, num_processes):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    next_frame_idx = 0
    received_frames = OrderedDict()

    finished_detect_process = 0

    while finished_detect_process < num_processes:
        if not result_queue.empty():
            frame_idx, frame = result_queue.get()
            if frame_idx is None:
                finished_detect_process += 1
                continue
            received_frames[frame_idx] = frame

            # 检查是否可以写入连续的帧
            while next_frame_idx in received_frames:
                writer.write(received_frames.pop(next_frame_idx))
                next_frame_idx += 1
                if next_frame_idx % 10 == 0:
                    print(f"已处理 {next_frame_idx}/{total_frames} 帧 ({next_frame_idx / total_frames * 100:.1f}%)")
        else:
            time.sleep(0.005)

    writer.release()
    print("写入进程结束")


def main(input_path,
         output_path='data/output/default_name.mp4',
         num_processes=4
         ):
    device = set_device()

    frame_queue = multiprocessing.Queue(maxsize=32)
    result_queue = multiprocessing.Queue()

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()  # 主进程中的cap可以释放了

    reader_process = multiprocessing.Process(
        target=read_frame,
        args=(input_path, frame_queue, 32, num_processes)
    )

    writer_process = multiprocessing.Process(
        target=write,
        args=(fps, width, height, total_frames, output_path, result_queue, num_processes)
    )
    detect_processes = [
        multiprocessing.Process(
            target=detect,
            args=(frame_queue, result_queue, device, i)
        )
        for i in range(num_processes)
    ]

    print("开始处理视频...")
    # 启动所有进程
    reader_process.start()
    for p in detect_processes:
        p.start()
    time.sleep(5)
    writer_process.start()

    # 等待所有进程完成
    reader_process.join()
    for p in detect_processes:
        p.join()
    writer_process.join()
    print("视频处理完成")

    cv2.destroyAllWindows()
