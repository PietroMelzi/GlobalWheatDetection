import pandas as pd
import numpy as np
import re
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader


DIR_INPUT = 'data'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'
DIR_WEIGHTS = 'data/weights'
WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet_fpn39.pth'


def get_model(num_classes=2):
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # num_classes = 2  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(WEIGHTS_FILE))
    model.eval()

    return model


class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).astype(np.uint8)
        # image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

# Albumentations
def get_test_transform():
    return A.Compose([
        A.CLAHE(clip_limit=3, p=1),
        # A.Resize(512, 512),
        A.Normalize(mean=0.0, std=1.0, p=1),
        ToTensorV2(p=1.0)
    ])

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def collate_fn(batch):
    return tuple(zip(*batch))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model()
model.to(device)

test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')
test_dataset = WheatTestDataset(test_df, DIR_TEST, get_test_transform())

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)

detection_threshold = 0.5
results = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for images, image_ids in test_data_loader:

    model = get_model()
    model = model.to(device)
    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)