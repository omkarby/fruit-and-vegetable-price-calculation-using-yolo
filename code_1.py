from ultralytics import YOLO
import os
import cv2
import json
import yaml

# --------- Helper: Load class names from data.yaml or coco.json ----------
def load_classes():
    if os.path.exists("data.yaml"):
        with open("data.yaml", "r") as f:
            data = yaml.safe_load(f)
        return data["names"]
    elif os.path.exists("_annotations.coco.json"):
        with open("_annotations.coco.json", "r") as f:
            data = json.load(f)
        # Extract unique category names
        return [c["name"] for c in data["categories"]]
    else:
        raise FileNotFoundError("No data.yaml or coco.json found!")

# --------- Helper: Load price list ----------
def load_prices(class_names):
    if os.path.exists("prices.json"):
        with open("prices.json", "r") as f:
            return json.load(f)
    # default fallback prices
    return {name: 50 for name in class_names}

# --------- Detection and billing ----------
def detect_and_bill(source_type):
    model = YOLO("best.pt")
    class_names = load_classes()
    prices = load_prices(class_names)

    if source_type == "image":
        path = input("Enter image path (or folder path): ").strip()
        results = model.predict(source=path, save=True, show=True)
        calc_bill(results, prices, class_names)

    elif source_type == "video":
        path = input("Enter video path: ").strip()
        results = model.predict(source=path, save=True, show=True)
        calc_bill(results, prices, class_names)

    elif source_type == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        print("🎥 Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, show=True, stream=True)
            for r in results:
                items = {}
                for box in r.boxes:
                    cls = int(box.cls)
                    label = class_names[cls]
                    items[label] = items.get(label, 0) + 1
                if items:
                    total = sum(prices.get(i, 0) * c for i, c in items.items())
                    print("Detected:", items, f"→ Bill: ₹{total}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

# --------- Billing summary ----------
def calc_bill(results, prices, class_names):
    all_items = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = class_names[cls]
            all_items[label] = all_items.get(label, 0) + 1

    print("\n🧾 Billing Summary")
    print("-" * 40)
    total = 0
    for item, count in all_items.items():
        price = prices.get(item, 0)
        subtotal = price * count
        total += subtotal
        print(f"{item:15} x{count:<2}  ₹{price:<5}  =  ₹{subtotal}")
    print("-" * 40)
    print(f"Total Bill: ₹{total}\n")

# --------- Main program ----------
if __name__ == "__main__":
    print("Select Input Type:")
    print("1️⃣  Image")
    print("2️⃣  Video")
    print("3️⃣  Webcam")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        detect_and_bill("image")
    elif choice == "2":
        detect_and_bill("video")
    elif choice == "3":
        detect_and_bill("webcam")
    else:
        print("Invalid choice.")

