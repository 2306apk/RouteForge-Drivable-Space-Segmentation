import cv2
import os

img_dir = "data/samples/CAM_FRONT"
mask_a = "masks/train_map_final"       # yours
mask_b = "masks/train_map_teammate"    # teammate

names = sorted(os.listdir(mask_a))[:20]

for name in names:
    img = cv2.imread(os.path.join(img_dir, name.replace(".png", ".jpg")))

    m1 = cv2.imread(os.path.join(mask_a, name), 0)
    m2 = cv2.imread(os.path.join(mask_b, name), 0)

    m1_col = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
    m2_col = cv2.applyColorMap(m2, cv2.COLORMAP_JET)

    overlay1 = cv2.addWeighted(img, 0.7, m1_col, 0.3, 0)
    overlay2 = cv2.addWeighted(img, 0.7, m2_col, 0.3, 0)

    combined = cv2.hconcat([overlay1, overlay2])

    cv2.imshow("YOURS (left) vs TEAMMATE (right)", combined)
    key = cv2.waitKey(0)

    if key == 27:
        break