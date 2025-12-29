from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")  # lightweight base model

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="object_removal_model"
    )

if __name__ == "__main__":
    train_model()
