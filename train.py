# 继承配置文件
_base_ = ['/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/config/mask_rcnn_r50_fpn_2x_coco.py']

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
 
# 修改数据配置
data = dict(
    train=dict(
        ann_file='/share12/home/zhenni/PythonProject/Dataset/balloon/annotations/train.json',
        img_prefix='/share12/home/zhenni/PythonProject/Dataset/balloon/train',
        classes=("balloon",)),
    val=dict(
        ann_file='/share12/home/zhenni/PythonProject/Dataset/balloon/annotations/val.json',
        img_prefix='/share12/home/zhenni/PythonProject/Dataset/balloon/val',
        classes=("balloon",))
)

# 修改模型
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1),
        mask_head=dict(
            num_classes=1)))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=8)
log_config = dict(interval=2, hooks=[dict(type='TextLoggerHook')])

load_from = '/share12/home/zhenni/PythonProject/openmmlab-camp/mmdet-1/ckpt/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'

