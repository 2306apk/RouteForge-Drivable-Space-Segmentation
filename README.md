MAHE Hackathon Project (Initial Commit From Arnav)

Rough Synopsis:

REAL TIME DRIVABLE SPACE SEGMENTATION

The car needs to know exactly which pixels on the screen are "road" and which are "not road" (sidewalks, grass, barriers).

Requirements: Familiarity with Encoder-Decoder models like U-Net or DeepLabV3+. You need to optimize for speed (FPS).

What you must produce:
A pipeline that processes a camera frame and spits out a binary mask (a black and white image where white = drivable road).
Edge Case Handling: Your model must not get confused by puddles (which look like sky) or gravel (which looks like off-road).

Success Metric: High mIoU (how well your "road" mask overlaps with the real road) while maintaining high FPS (frames per second).