import cv2
from ultralytics import YOLO


model = YOLO(r"D:\Python\Projects\Community\School\DAT\SeaDroneSee\best.pt")

# Open video source (0 for webcam or "path/to/video.mp4")
results = model.track(source=r"D:\Python\Projects\Community\School\DAT\SeaDroneSee\test.mp4", stream=True, tracker="bytetrack.yaml")

for r in results:
    img = r.orig_img
    
    for box in r.boxes:
        # Get coordinates, confidence, and class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Label mapping
        label_name = "Human" if cls == 1 else "Object"
        display_text = f"{label_name} {conf:.2f}"
        
        # Draw Bounding Box
        color = (0, 255, 0) if cls == 1 else (255, 0, 0) # Green for Human, Blue for Object
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label & Confidence
        cv2.putText(img, display_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("SAR Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

