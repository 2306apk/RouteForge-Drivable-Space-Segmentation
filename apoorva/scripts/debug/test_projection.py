import cv2
import matplotlib.pyplot as plt
from nuscenes_loader import load_all_metadata, get_cam_front_frames
from geometry import project_point

meta = load_all_metadata()
frames = get_cam_front_frames(meta)

frame = frames[0]

img = cv2.imread(frame["image_path"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calib = meta["calibrated_sensor"][frame["calibrated_sensor_token"]]
ego = meta["ego_pose"][frame["ego_pose_token"]]

# 🔥 TEST POINT (straight ahead on road)
point_world = [
    ego["translation"][0],
    ego["translation"][1] + 5,  # forward
    ego["translation"][2]
]

pixel = project_point(point_world, ego, calib)

if pixel:
    u, v = pixel
    cv2.circle(img, (u, v), 8, (255, 0, 0), -1)

plt.imshow(img)
plt.title("Projection Test")
plt.axis("off")
plt.show()