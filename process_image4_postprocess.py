import os
import numpy as np
from PIL import Image

# 设置文件夹路径
image_folder = '  '
mask_folder = '  '
output_folder = '  '

# 获取影像名称列表（不含后缀）
img_file_names = [f.split('.')[0] for f in os.listdir(image_folder) if f.endswith('.tif')]

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

for img_file_name in img_file_names:
    # 构造影像文件路径
    region_file = os.path.join(mask_folder, img_file_name + 'mask_t.tif')
    boundary_file = os.path.join(mask_folder, img_file_name + 'boundary_t.tif')
    
    # 检查文件是否存在
    if os.path.exists(region_file) and os.path.exists(boundary_file):
        # 读取地块内部和地块边界结果影像
        region_image = np.array(Image.open(region_file))
        boundary_image = np.array(Image.open(boundary_file))
        
        # 修改地块内部结果：像素值为1的变为像素值为2
        image_region_1 = np.where(region_image == 1, 2, region_image)
        
        # 获取边界结果中像素值为1的位置
        boundary_coords = np.argwhere(boundary_image == 1)
        
        # 将对应位置的像素值在地块内部结果影像中修改为1
        for coord in boundary_coords:
            image_region_1[coord[0], coord[1]] = 1
        
        # 保存结果
        output_file = os.path.join(output_folder, img_file_name + '.tif')
        Image.fromarray(image_region_1).save(output_file)

    else:
        print(f"文件 {img_file_name} 对应的地块内部或边界结果不存在，跳过。")

print("处理完成。")
