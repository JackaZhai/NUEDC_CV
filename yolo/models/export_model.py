from ultralytics import YOLO

pt = r"best.pt"
input_width = 320
input_height = 224
model = YOLO(pt)
onnx_path = model.export(
    format="onnx",
    imgsz=(input_height, input_width),   # H, W
    dynamic=False,
    simplify=True, 
    opset=17,
)
print("ONNX saved to:", onnx_path)
