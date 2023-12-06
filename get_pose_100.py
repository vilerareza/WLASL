import os
import shutil

split_path = 'E:/buffer/asl100.json'
source_dir = 'E:/buffer/pose_per_individual_videos/pose_per_individual_videos'
target_dir = 'pose_100/'

with open (split_path, 'r') as file:
    content = file.readlines()

content = ''.join(content)
content = eval(content)

video_ids = []

for i in content:
    for j in i['instances']:
        video_id =  (j['video_id'])
        video_ids.append(video_id)
        shutil.copytree(f'{source_dir}/{video_id}', f'{target_dir}/{video_id}')
