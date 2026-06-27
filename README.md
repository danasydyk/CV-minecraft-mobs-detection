# Minecraft Mobs Detection - YOLOv8

A real-time object detection model trained to detect Minecraft hostile mobs using YOLOv8.

![Demo](https://github.com/user-attachments/assets/ebea59c6-7260-4530-8f64-2c9f35224a06)

##  Live Demo

 [Try it on Hugging Face](https://huggingface.co/spaces/coolbambook/minecraft-mobs-detector)

Upload a Minecraft screenshot or video and the model will detect and label all visible mobs.

##  Results

| Metric | Score |
|--------|-------|
| mAP50 | 0.95 |
| mAP50-95 | 0.77 |

##  Classes

| ID | Class | Includes |
|----|-------|---------|
| 0 | Creeper | Standard Creeper |
| 1 | Skeleton | Standard, Wither, Bogged, Stray |
| 2 | Spider | Standard, Cave Spider |
| 3 | Zombie | Standard, Drowned, Husk |
| 4 | Enderman | Standard Enderman |

##  Dataset

[Minecraft Mobs YOLO Dataset](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset) — 1,972 images, pre-split 80/20 train/val, includes ~24% background images for hard negative mining.

##  Training

- **Model:** YOLOv8n (nano)
- **Epochs:** 50
- **Image size:** 640px
- **Batch size:** 16
- **Framework:** Ultralytics YOLOv8

##  Run Locally

```bash
pip install ultralytics gradio opencv-python
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.
