from .coco import CocoDetection
from .transforms.ctdet import CTTransform

transform_factory = {
    'ctdet': CTTransform
}


def get_dataset(dataset, task, split):
    if dataset == 'coco':
        if split == 'train':
            return CocoDetection(
                root='../data/coco/train2017',
                annFile='../data/coco/annotations/instances_train2017.json',
                transforms=transform_factory[task](split)
            )
        elif split == 'val':
            return CocoDetection(
                root='../data/coco/val2017',
                annFile='../data/coco/annotations/instances_val2017.json',
                transforms=transform_factory[task](split)
            )
    return None
