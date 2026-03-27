import cv2
import torch
from training.model import UNet

model = UNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (256, 256)) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(img)[0][0].numpy()

    mask = (pred > 0.5).astype("uint8") * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    cv2.imshow("Drivable Space", mask)

    if cv2.waitKey(1) == 27:
        break