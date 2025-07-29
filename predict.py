from ultralytics import YOLOv10

# Load a pretrained YOLOv10n model
model = YOLOv10("./runs/detect/train6/weights/best.pt")

# Perform object detection on an image
# results = model("test1.jpg")
results = model.predict("D:/Develop/Pascal Voc/pascal-voc/VOCdevkit/images/train/2007_000129.jpg")

# Display the results
results[0].show()