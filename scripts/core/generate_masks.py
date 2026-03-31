import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LOAD MODEL (MATCH TRAINING)
# ------------------------
model = UNet().to(device)

model.load_state_dict(torch.load("/content/model.pth", map_location=device))
model.eval()

# ------------------------
# FIX IMAGE PATH - SET YOUR EDGE-CASE IMAGE HERE!
# ------------------------
# Replace this with the path to your specific edge-case image
img_path = "/content/Dataset/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402944662460.jpg" # <--- ADD YOUR IMAGE NAME HERE, e.g., "your_grass_puddle_image.jpg"

print("Using image:", img_path)

# ------------------------
# LOAD IMAGE
# ------------------------
img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"Image not found at {img_path}")

orig = img.copy()

# ------------------------
# PREPROCESS
# ------------------------
# Revert to original training resolution for inference to ensure correctness
img_resized = cv2.resize(img, (128, 128)) # Increased resolution for potentially better quality
img_resized = img_resized / 255.0
img_resized = np.transpose(img_resized, (2, 0, 1))

img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------------
# FPS TEST (can be removed if not needed)
# ------------------------
start = time.time()

for _ in range(100):
    with torch.no_grad():
        pred = model(img_tensor)

end = time.time()
fps = 100 / (end - start)

print("FPS:", fps)

# ------------------------
# PREDICTION
# ------------------------
with torch.no_grad():
    pred = model(img_tensor)

mask = torch.sigmoid(pred).cpu().numpy()[0, 0]
mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

mask_bin = (mask > 0.5).astype(np.uint8)

# ------------------------
# POST-PROCESSING (to block out noise)
# ------------------------
# Define a kernel for morphological operations. A 3x3 or 5x5 kernel is common.
kernel = np.ones((5,5),np.uint8)

# Apply morphological opening: Erosion followed by Dilation.
# This removes small, isolated noise (false positives) from the binary mask.
mask_bin_processed = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

# You can also apply morphological closing (dilation then erosion) to fill small holes:
# mask_bin_processed = cv2.morphologyEx(mask_bin_processed, cv2.MORPH_CLOSE, kernel)

# ------------------------
# OVERLAY
# ------------------------
overlay = orig.copy()
# Use the processed mask for the overlay visualization
overlay[mask_bin_processed == 1] = [0, 255, 0] # Green overlay for drivable areas

# ------------------------
# DISPLAY (COLAB SAFE)
# ------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Processed Mask") # Updated title to reflect post-processing
plt.imshow(mask_bin_processed, cmap='gray')

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

plt.show()
