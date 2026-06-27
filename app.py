import tempfile
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile, os

model = YOLO("best.pt")

CLASSES = ["creeper", "skeleton", "spider", "zombie", "enderman"]
COLORS = {
    "creeper":  (0, 200, 0),
    "skeleton": (200, 200, 200),
    "spider":   (180, 0, 180),
    "zombie":   (0, 180, 80),
    "enderman": (80, 80, 80),
}

def draw_boxes(img, results):
    
    for box in results.boxes:
        cls  = CLASSES[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = COLORS[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls} {conf:.0%}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def detect_image(image):
    results = model.predict(image, conf=0.25, verbose=False)[0]
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = draw_boxes(img, results)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = [f"{CLASSES[int(b.cls)]} ({float(b.conf):.0%})" for b in results.boxes]
    label = "\n".join(detections) if detections else "No mobs detected."
    return img, label


def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25, verbose=False)[0]
        frame = draw_boxes(frame, results)
        out.write(frame)

    cap.release()
    out.release()
    return out_path



with gr.Blocks(title="Minecraft Mobs Detector") as demo:
    gr.Markdown("# Minecraft Mobs Detector")
    gr.Markdown("Upload a **screenshot** or a **video** to detect mobs: Creeper, Skeleton, Spider, Zombie, Enderman.")

    with gr.Tab("Image"):
        with gr.Row():
            img_input  = gr.Image(type="pil", label="Upload Screenshot")
            img_output = gr.Image(type="numpy", label="Detections")
        txt_output = gr.Textbox(label="Detected Mobs")
        gr.Button("Detect").click(detect_image, inputs=img_input, outputs=[img_output, txt_output])

    with gr.Tab("Video"):
        vid_input  = gr.Video(label="Upload Video")
        vid_output = gr.Video(label="Annotated Video")
        gr.Button("Detect").click(detect_video, inputs=vid_input, outputs=vid_output)

demo.launch(allowed_paths=[tempfile.gettempdir()])
