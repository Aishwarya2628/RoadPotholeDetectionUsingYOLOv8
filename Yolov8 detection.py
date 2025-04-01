from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from geopy.geocoders import Nominatim


DEFAULT_LATITUDE = 9.6734
DEFAULT_LONGITUDE = 78.0995

# Function to get location from GPS coordinates
def get_location(lat, lon):
    geolocator = Nominatim(user_agent="pothole_detector")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        return location.address if location else "Unknown Location"
    except Exception as e:
        print(f"Error fetching location: {e}")
        return "Unknown Location"

# Function to send email
def send_email(subject, body):
    sender_email = "kaishwarya978@gmail.com"
    receiver_email = "kppavithra4429@gmail.com"
    password = "njdz qgxy zuqn vyvk" 

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"Error: {e}")

# Function to create message with pothole data and GPS location
def create_message(small_count, medium_count, large_count, location):
    message = f"Pothole Detection Summary:\n\n"
    message += f"📍 Location: {location}\n\n"
    message += f"Small Potholes: {small_count}\n"
    message += f"Medium Potholes: {medium_count}\n"
    message += f"Large Potholes: {large_count}\n"
    return message

# Initialize the YOLO model
model = YOLO("best.pt")

# Open the video file
cap = cv2.VideoCapture('p.mp4')
frame_count = 0

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object for output
output_video = cv2.VideoWriter(
    'output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height)
)

# Thresholds for pothole size classification
SMALL_SIZE_THRESHOLD = 200
MEDIUM_SIZE_THRESHOLD = 3500

# Tracking potholes
pothole_tracker = defaultdict(dict)
next_pothole_id = 0
detected_pothole_ids = set()

# Function to classify pothole size
def classify_size(area):
    if area < SMALL_SIZE_THRESHOLD:
        return "Small", (0, 255, 0)  # Green
    elif area < MEDIUM_SIZE_THRESHOLD:
        return "Medium", (255, 255, 0)  # Yellow
    else:
        return "Large", (0, 0, 255)  # Red

# Pothole counters
small_count = 0
medium_count = 0
large_count = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("End of video reached.")
        break
    frame_count += 1

    img = cv2.resize(img, (frame_width, frame_height))
    h, w, _ = img.shape

    # Perform detection
    results = model.predict(img, conf=0.5)
    potholes_in_frame = []

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu().numpy()
            for seg, box in zip(masks, boxes):
                seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(seg_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w1, h1 = cv2.boundingRect(contour)
                    centroid = (int(x + w1 / 2), int(y + h1 / 2))

                    pothole_found = False
                    for pothole_id, pothole_data_item in pothole_tracker.items():
                        px, py, p_area, p_size, p_color = pothole_data_item.values()

                        if np.linalg.norm(np.array(centroid) - np.array((px, py))) < 50:
                            pothole_tracker[pothole_id]["x"] = centroid[0]
                            pothole_tracker[pothole_id]["y"] = centroid[1]
                            pothole_tracker[pothole_id]["area"] = area

                            cv2.rectangle(img, (x, y), (x + w1, y + h1), p_color, 2)
                            cv2.putText(img, f"{p_size} Pothole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2)

                            potholes_in_frame.append(pothole_id)
                            pothole_found = True
                            break

                    if not pothole_found:
                        size_label, color = classify_size(area)
                        pothole_tracker[next_pothole_id] = {
                            "x": centroid[0],
                            "y": centroid[1],
                            "area": area,
                            "size": size_label,
                            "color": color
                        }
                        potholes_in_frame.append(next_pothole_id)

                        if next_pothole_id not in detected_pothole_ids:
                            detected_pothole_ids.add(next_pothole_id)
                            if size_label == "Small":
                                small_count += 1
                            elif size_label == "Medium":
                                medium_count += 1
                            else:
                                large_count += 1

                        cv2.rectangle(img, (x, y), (x + w1, y + h1), color, 2)
                        cv2.putText(img, f"{size_label} Pothole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        next_pothole_id += 1

    for pothole_id in list(pothole_tracker.keys()):
        if pothole_id not in potholes_in_frame:
            del pothole_tracker[pothole_id]

    output_video.write(img)
    cv2.imshow('Pothole Detection', img)
    cv2.waitKey(int(1000 / fps))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Get location from GPS coordinates
pothole_location = get_location(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)

# Generate report with location
message = create_message(small_count, medium_count, large_count, pothole_location)

# Send email with pothole report
send_email("Pothole Detection Report", message)
