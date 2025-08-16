import os
import cv2
import numpy as np
import torch
from utils import get_frames, get_batches, predictions_to_scenes

# ====== CHỈNH Ở ĐÂY ======
VIDEO_DIR = "../videos"              # thư mục chứa video
MODEL_WEIGHTS = "ckpt_0_200_0.pth"   # đường dẫn file pretrained
OUTPUT_DIR = "./scenes_csv"          # thư mục lưu CSV
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.296   # ngưỡng mặc định của repo
# ========================

if __name__ == "__main__":
    # Import model
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet
    model = TransNetV2Supernet().eval()

    # Load trọng số
    if os.path.exists(MODEL_WEIGHTS):
        print(f"Loading model weights from {MODEL_WEIGHTS}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise FileNotFoundError("Không tìm thấy file model!")

    if DEVICE == "cuda":
        model = model.cuda(0)
    model.eval()

    def predict(batch):
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float()
        batch = batch.to(DEVICE)
        one_hot = model(batch)
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        return torch.sigmoid(one_hot[0])

    # tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # duyệt tất cả video trong folder
    for file_name in os.listdir(VIDEO_DIR):
        if not file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = os.path.join(VIDEO_DIR, file_name)
        print(f"\n=== Xử lý video: {video_path} ===")

        # Đọc video
        frames = get_frames(video_path)

        # Chạy model theo batch/windows
        predictions_parts = []
        for batch in get_batches(frames):
            one_hot = predict(batch)
            one_hot = one_hot.detach().cpu().numpy()
            predictions_parts.append(one_hot[25:75])

        predictions = np.concatenate(predictions_parts, 0)[:len(frames)]
        bin_pred = (predictions > THRESH).astype(np.uint8)
        scenes = predictions_to_scenes(bin_pred)

        print(f"Tổng số cảnh: {len(scenes)}")
        for idx, (start, end) in enumerate(scenes):
            print(f"Cảnh {idx+1}: Frame {start} -> {end}")

        # Lưu CSV cho từng video
        base_name = os.path.splitext(file_name)[0]
        out_csv = os.path.join(OUTPUT_DIR, base_name + "_scenes.csv")
        with open(out_csv, "w") as f:
            f.write("scene_idx,start_frame,end_frame\n")
            for i, (s, e) in enumerate(scenes):
                f.write(f"{i+1},{int(s)},{int(e)}\n")
        print("Saved scenes CSV:", out_csv)
