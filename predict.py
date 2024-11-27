# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog
import time
import subprocess
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import tempfile
import shutil
from cog import BasePredictor, Input, Path

# Add the sam2 module to the Python path
sys.path.append("./sam2")

DEVICE = "cuda"
MODEL_CACHE = "sam2/checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/sam-2/checkpoints/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

class Predictor(BasePredictor):
    def setup(self) -> None:
        global build_sam2_video_predictor

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print("sam2 not found. Installing...")
            os.system("pip install --no-build-isolation -e .")
            from sam2.build_sam import build_sam2_video_predictor

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = ["sam2.1_hiera_base_plus.pt"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
        
        # Ensure output directory exists
        os.makedirs("visualization/samurai/base_plus", exist_ok=True)
        
        # Set the model path (using same default as demo.py)
        self.model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        self.model_cfg = determine_model_cfg(self.model_path)
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.model_path, device=DEVICE)

    def predict(
        self,
        video: Path = Input(description="Input video to process"),
        x_coordinate: int = Input(description="x-coordinate of top-left corner of bounding box", default=100),
        y_coordinate: int = Input(description="y-coordinate of top-left corner of bounding box", default=100),
        width: int = Input(description="Width of bounding box", default=400),
        height: int = Input(description="Height of bounding box", default=300),
    ) -> Path:
        """Run object tracking on the input video with the specified bounding box"""
        try:
            # Prepare video input
            frames_or_path = prepare_frames_or_path(str(video))
            bbox = (x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height)
            prompts = {0: (bbox, 0)}  # Initialize with first frame bbox
            color = [(255, 0, 0)]  # Red color for visualization
            
            # Create temporary directory for frame processing
            temp_dir = tempfile.mkdtemp()
            frame_count = 0

            print("[*] Loading video frames...")
            # Load video frames
            if osp.isdir(str(video)):
                # Handle directory of frames
                frames = sorted([osp.join(str(video), f) for f in os.listdir(str(video)) if f.endswith(".jpg")])
                loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
                frame_height, frame_width = loaded_frames[0].shape[:2]
                fps = 30  # Default fps for image sequence
            else:
                # Handle video file
                cap = cv2.VideoCapture(str(video))
                loaded_frames = []
                fps = cap.get(cv2.CAP_PROP_FPS)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    loaded_frames.append(frame)
                cap.release()
                if len(loaded_frames) == 0:
                    raise ValueError("No frames were loaded from the video.")
                frame_height, frame_width = loaded_frames[0].shape[:2]

            num_frames = len(loaded_frames)
            print(f"[+] Loaded {num_frames} frames")

            print("[*] Processing frames with model...")
            # Process video with model
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16):
                # Initialize state with first frame
                state = self.predictor.init_state(frames_or_path, offload_video_to_cpu=True)
                bbox, track_label = prompts[0]
                _, _, masks = self.predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

                # Process each frame
                for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                    # Check if frame_idx is valid
                    if frame_idx >= num_frames:
                        print(f"[!] Warning: frame_idx {frame_idx} exceeds number of loaded frames {num_frames}")
                        continue

                    mask_to_vis = {}
                    bbox_to_vis = {}

                    # Process each object in the frame
                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].cpu().numpy()
                        mask = mask > 0.0
                        non_zero_indices = np.argwhere(mask)
                        if len(non_zero_indices) == 0:
                            bbox = [0, 0, 0, 0]
                        else:
                            y_min, x_min = non_zero_indices.min(axis=0).tolist()
                            y_max, x_max = non_zero_indices.max(axis=0).tolist()
                            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        bbox_to_vis[obj_id] = bbox
                        mask_to_vis[obj_id] = mask

                    # Visualize results on frame
                    img = loaded_frames[frame_idx].copy()
                    
                    # Draw masks
                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                    # Draw bounding boxes
                    for obj_id, bbox in bbox_to_vis.items():
                        cv2.rectangle(img, 
                                    (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                    color[obj_id % len(color)], 2)

                    # Save processed frame
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:05d}.png")
                    cv2.imwrite(frame_path, img)
                    frame_count += 1

            print("[*] Generating output video...")
            # Generate output video
            video_name = os.path.basename(str(video))
            output_path = f"visualization/samurai/base_plus/{video_name}"
            frames_pattern = os.path.join(temp_dir, "frame_%05d.png")
            ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frames_pattern} -c:v libx264 -pix_fmt yuv420p {output_path}"
            os.system(ffmpeg_cmd)
            
            print("[+] Cleaning up...")
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

            return Path(output_path)

        except Exception as e:
            print(f"[ERROR] An error occurred: {str(e)}")
            # Clean up on error
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
            raise e