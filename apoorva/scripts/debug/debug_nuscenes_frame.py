import cv2
import matplotlib.pyplot as plt
from nuscenes_loader import load_all_metadata, get_cam_front_frames

meta = load_all_metadata()
frames = get_cam_front_frames(meta)

# Pick one frame
frame = frames[0]

img = cv2.imread(frame["image_path"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("CAM_FRONT Frame")
plt.axis("off")
plt.show()

print("\nFrame Info:")
print(frame)