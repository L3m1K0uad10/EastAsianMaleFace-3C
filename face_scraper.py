import os
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO
import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize face detector
detector = MTCNN()

# Directories
INPUT_DIR = "raw_images"
OUTPUT_DIR = "processed_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API Key for SerpAPI (you need to sign up for this)
API_KEY = 'your_serpapi_key_here'

# Search query for male faces from East Asian countries
SEARCH_QUERIES = ["Japanese male face", "Chinese male face", "Korean male face"]

# Image size to save after processing
IMAGE_SIZE = (160, 160)  # Resized image size

# Function to scrape images using SerpAPI
def scrape_images(query, num_images=20):
    params = {
        "q": query,
        "tbm": "isch",  # Image search
        "api_key": API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    image_urls = []
    for image in results.get("images_results", []):
        if "https" in image.get("url", ""):
            image_urls.append(image["url"])

    return image_urls[:num_images]

# Function to process and crop the face
def process_face(image, output_filename):
    # Detect faces in the image
    result = detector.detect_faces(image)
    
    if result:
        for i, face in enumerate(result):
            try:
                face_img = crop_face_strict(image, face['keypoints'])
                output_path = os.path.join(OUTPUT_DIR, f"{output_filename}_face{i}.jpg")
                cv2.imwrite(output_path, face_img)
                print(f"Saved face image: {output_path}")
            except Exception as e:
                print(f"Error processing {output_filename}: {e}")
    else:
        print(f"No face found in {output_filename}")

# Cropping function (based on landmarks: eyes, nose, mouth)
def crop_face_strict(image, landmarks, margin=0.2):
    # Get coordinates of key points (eyes, nose, mouth)
    points = np.array([landmarks['left_eye'], landmarks['right_eye'],
                       landmarks['nose'], landmarks['mouth_left'], landmarks['mouth_right']])
    
    # Determine bounding box around the face
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    # Add margin around the bounding box
    box_size = max(x_max - x_min, y_max - y_min) * (1 + margin)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate new coordinates for cropping
    x1 = int(center_x - box_size / 2)
    y1 = int(center_y - box_size / 2)
    x2 = int(center_x + box_size / 2)
    y2 = int(center_y + box_size / 2)

    # Clip to image boundaries
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop the image and resize to target size
    face = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face, IMAGE_SIZE)
    return face_resized

# Main function to scrape and process faces
def scrape_and_process():
    for query in SEARCH_QUERIES:
        print(f"Scraping for query: {query}")
        image_urls = scrape_images(query)

        for idx, url in enumerate(image_urls):
            try:
                # Download the image
                print(f"Downloading image {idx+1}: {url}")
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image_cv = np.array(image)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
                
                # Process the image and save the faces
                process_face(image_cv, f"{query.replace(' ', '_')}_{idx+1}")
            except Exception as e:
                print(f"Failed to download or process image {idx+1} from {url}: {e}")

if __name__ == "__main__":
    scrape_and_process()
