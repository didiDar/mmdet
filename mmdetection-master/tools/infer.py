import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

# 指定模型的配置文件和 checkpoint 文件路径
config_file = './test_results/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint_file = './test_results/latest.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并展示结果
img = ('test.jpg')  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
result = inference_detector(model, img)

# 输出标注信息
print(result)
# 在一个新的窗口中将结果可视化
show_result_pyplot(model, img, result)
# 或者将可视化结果保存为图片
model.show_result(img, result, out_file='result.jpg')
