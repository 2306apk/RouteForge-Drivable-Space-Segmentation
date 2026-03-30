import cv2
import os

img_dir = "data/samples/CAM_FRONT"
mask_dir = "masks/train_map_final"

for name in os.listdir(mask_dir)[:10]:
    mask = cv2.imread(os.path.join(mask_dir, name), 0)
    img = cv2.imread(os.path.join(img_dir, name.replace(".png", ".jpg")))

    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)