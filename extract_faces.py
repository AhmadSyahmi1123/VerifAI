# pip install mediapipe opencv-python tqdm
import cv2
import mediapipe as mp
import os
from tqdm import tqdm

mp_face_detection = mp.solutions.face_detection

def extract_faces_mediapipe(video_path, output_folder, size=(224,224), every_n_frames=5, min_detection_confidence=0.5):
    """
    Extract faces from a video using MediaPipe Face Detection.
    Args:
        video_path (str): Path to the video.
        output_folder (str): Folder to save extracted face images.
        size (tuple): Resize faces to this size.
        every_n_frames (int): Sample every N frames.
        min_detection_confidence (float): Threshold for face detection confidence.
    Returns:
        int: Number of faces saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_detection_confidence) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        x, y = max(0, x), max(0, y)
                        face = frame[y:y+h_box, x:x+w_box]
                        if face.size == 0:
                            continue  # skip empty crops
                        face = cv2.resize(face, size)
                        out_path = os.path.join(output_folder, f"frame{frame_idx}_face{i}.jpg")
                        cv2.imwrite(out_path, face)
                        saved += 1
            frame_idx += 1

    cap.release()
    return saved

def process_dataset(input_root="datasets", output_root="processed", size=(224,224), every_n_frames=5):
    """
    Process all videos in the dataset folder and extract faces.
    Folder structure expected:
    dataset/
        real/
        fake/
    """
    for cls in ["real", "fake"]:
        in_dir = os.path.join(input_root, cls)
        out_dir_cls = os.path.join(output_root, cls)
        os.makedirs(out_dir_cls, exist_ok=True)

        video_list = [v for v in os.listdir(in_dir) if v.endswith((".mp4", ".avi", ".mov"))]
        for video in tqdm(video_list, desc=f"Processing {cls} videos"):
            video_path = os.path.join(in_dir, video)
            video_name = os.path.splitext(video)[0]
            out_dir_video = os.path.join(out_dir_cls, video_name)
            count = extract_faces_mediapipe(video_path, out_dir_video, size=size, every_n_frames=every_n_frames)
            # Optional: print progress per video
            # print(f"Saved {count} faces from {video}")

if __name__ == "__main__":
    process_dataset(input_root="datasets", output_root="processed", size=(224,224), every_n_frames=5)
    print("✅ All videos processed and faces extracted!")
