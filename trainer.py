import os
import cv2
import numpy as np
from PIL import Image
import cv2.face

# Create a face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "datasets"

def get_images_with_id(path):
    # Get all file paths in the given directory
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_paths:
        # Open image and convert to grayscale
        faceImg = Image.open(single_image_path).convert('L')  # 'L' mode = luminance, converts to grayscale
        # Convert PIL image to numpy array
        faceNp = np.array(faceImg, np.uint8)
        # Extract ID from filename (assumes filename format: name.id.extension)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        # Append face array and id to respective lists
        faces.append(faceNp)
        ids.append(id)
        # Display the training image
        cv2.imshow("TRAINING", faceNp)
        cv2.waitKey(100)  # Wait for 100ms between displaying images

    return np.array(ids), faces

# Get the faces and IDs
ids, faces = get_images_with_id(path)
# Train the face recognizer
recognizer.train(faces, ids)  # Corrected: changed 'id' to 'ids'
# Save the trained model
recognizer.save("recognizer/trainingdata.yml")
# Close all OpenCV windows
cv2.destroyAllWindows()
