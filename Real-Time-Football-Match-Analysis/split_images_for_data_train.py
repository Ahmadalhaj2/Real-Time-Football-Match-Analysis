import av
import os
import torch
from torchvision.transforms.functional import to_pil_image

def extract_frames_to_gpu(video_path, output_folder, interval_sec=3, device="cuda"):
    os.makedirs(output_folder, exist_ok=True)

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = "cpu"
    else:
        print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    interval_pts = int(fps * interval_sec)

    print(f"Video FPS: {fps}")
    print(f"Extracting 1 frame every {interval_sec} seconds")
    print(f"Moving frames to: {device.upper()}")

    saved = 0
    for i, frame in enumerate(container.decode(video=0)):
        if frame.pts is None:
            continue
        if i % interval_pts != 0:
            continue

        img_array = frame.to_ndarray(format='rgb24')
        img_tensor = torch.tensor(img_array, dtype=torch.uint8).permute(2, 0, 1).to(device)

        print(f"Frame {i} loaded to device: {img_tensor.device}")

        # Optional dummy GPU op to trigger some GPU activity
        _ = img_tensor.float().mean()

        img = to_pil_image(img_tensor.cpu())
        img.save(os.path.join(output_folder, f"frame_{saved:04d}.jpg"))
        saved += 1

    print("Done.")


extract_frames_to_gpu("Dailymotion.mp4", "data", interval_sec=7, device="cuda")
