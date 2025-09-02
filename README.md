## **Dataset Description: East Asian Male Face Dataset**

### **Name**: `EastAsianMaleFace-3C`

### **Description**:

The `EastAsianMaleFace-3C` dataset is a curated image collection focused on male facial images from three East Asian nationalities: Chinese, Japanese, and Korean. The dataset is intended for training machine learning models to classify facial features by national origin, with an emphasis on reducing noise such as hair, background, and accessories by strictly cropping to facial features (eyes, nose, mouth, ears).

### **Purpose**:

To support research and experimentation in:

* Nationality classification from facial features

* Bias analysis in facial recognition

* Cultural and regional visual feature learning

* Lightweight classification pipelines using cropped, normalized face images

### **Structure**:

EastAsianMaleFace-3C/

├── Chinese/

│   ├── img001.jpg

│   ├── ...

├── Japanese/

│   ├── img101.jpg

├── Korean/

│   ├── img201.jpg

* `Classes`: Chinese, Japanese, Korean

* `Gender`: Male only

* `Age range`: ~18–50 (varied across sources)

* `Format`: .jpg, RGB, 160x160 or 224x224 pixels (cropped and resized)

* `Face Region`: Strictly cropped to include only eyes, nose, mouth, and ears. Excludes hair, chin, neck, and background.

### **Image Source**:

Images are sourced from:

* Publicly available media (news, Wikipedia, public celebrity images)

* Scraped and filtered using facial detection models (MTCNN)

* Manually verified for gender, clarity, and nationality accuracy

⚠️ Dataset is under construction and may be refined with additional metadata (age, emotion, face angles) in future versions.

#### **Data Collection**

**Image Scraper**

You can use the provided face_scraper.py script to automatically download images of male faces from three East Asian nationalities: Chinese, Japanese, and Korean. The scraper pulls images using SerpAPI and saves them into the data/raw_images/ folder.

### **Preproccesiong**:

Each image has been:

* Face-detected and landmark-localized using `MTCNN`

* Cropped tightly around key facial features (not full head)

* Resized to uniform resolution (e.g., 160x160)

* Optionally aligned (in future versions)

### Licence & Usage

* This dataset is for research and educational purposes only.

* Respect privacy and do not use for commercial facial recognition applications.

* Attribution required when using in publications or derivative work.
