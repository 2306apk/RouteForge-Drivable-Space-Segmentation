from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-mini',
    dataroot='Dataset',
    verbose=True
)

print("Total samples:", len(nusc.sample))

#Extracting images
sample = nusc.sample[0]
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
img_path = nusc.get_sample_data_path(cam_data['token'])
print(img_path)