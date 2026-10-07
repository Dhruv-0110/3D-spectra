import cv2
import numpy as np
import os
video_path = "input_video.mp4"

# Folder where sharp keyframes will be saved
output_folder = "keyframes"
os.makedirs(output_folder, exist_ok=True)

# Blur detection threshold
threshold = 50

# Open the input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video could not be opened.")
    exit()
frame_number = 0
saved_frames = 0

min_score = float("inf")
max_score = float("-inf")
total_score = 0

print("Video processing started...\n")

# Read and process the video frame by frame
while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Laplacian operator to detect intensity changes and edges
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian
    blur_score = np.var(laplacian)

    min_score = min(min_score, blur_score)
    max_score = max(max_score, blur_score)
    total_score += blur_score

    # Compare the blur score with the threshold
    if blur_score >= threshold:
        saved_frames += 1
        filename = f"frame_{saved_frames:04d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        print(
            f"Frame {frame_number}: "
            f"Score = {blur_score:.2f} → SAVED"
        )
    else:
        print(
            f"Frame {frame_number}: "
            f"Score = {blur_score:.2f} → DISCARDED"
        )
cap.release()
print("\n-----------------------------")
print("Processing completed!")
print("Total frames:", frame_number)
print("Sharp frames saved:", saved_frames)
print("Blurry frames discarded:", frame_number - saved_frames)
print("-----------------------------")

average_score = total_score / frame_number

print("Minimum score:", min_score)
print("Maximum score:", max_score)
print("Average score:", average_score)