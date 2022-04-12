import os.path
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image

from torchvision.datasets.vision import VisionDataset


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def load_image(self, index: int) -> Tuple[int, Image.Image]:
        path = self.coco.loadImgs(self.ids[index])[0]["file_name"]
        return self.ids[index], Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

    def run_eval(self, results, save_dir):
        from pycocotools.cocoeval import COCOeval
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def save_results(self, results, save_dir):
        import json
        detections = []
        for image_id in results:
            for cls_ind in results[image_id]:
                for bbox in results[image_id][cls_ind]:
                    score = round(bbox[4].item(), 3)
                    bbox_out = [round(t,3) for t in bbox[0:4].tolist()]
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(cls_ind),
                        "bbox": bbox_out,
                        "score": score
                    }
                    if len(bbox) > 5:
                        extreme_points = bbox[5:13].tolist()
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        json.dump(detections, open('{}/results.json'.format(save_dir), 'w'))