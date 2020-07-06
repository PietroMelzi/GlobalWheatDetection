import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations
import re


TRAIN_DIR = "data/train/"


def get_bbox(bboxes, col, color='white', bbox_format='pascal_voc'):
    for i in range(len(bboxes)):
        # Create a Rectangle patch
        if bbox_format == 'pascal_voc':
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2] - bboxes[i][0],
                bboxes[i][3] - bboxes[i][1],
                linewidth=2,
                edgecolor=color,
                facecolor='none')
        else:
            rect = patches.Rectangle(
                (bboxes[i][0], bboxes[i][1]),
                bboxes[i][2],
                bboxes[i][3],
                linewidth=2,
                edgecolor=color,
                facecolor='none')

        # Add the patch to the Axes
        col.add_patch(rect)


def augmentation(train_df, img_id):

    image = cv2.imread(TRAIN_DIR + img_id + '.jpg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    # image /= 255.0

    records = train_df[train_df['image_id'] == img_id]
    boxes = records[['x', 'y', 'w', 'h']].values
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    labels = np.ones((len(boxes),))

    aug = albumentations.Compose([
        albumentations.RandomSizedCrop(min_max_height=(768, 768), height=1024, width=1024)
        # albumentations.CLAHE(p=1, clip_limit=1),
        # albumentations.Normalize(mean=0.0, std=1.0, p=1)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    aug_result = aug(image=image, bboxes=boxes, labels=labels)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
    get_bbox(boxes, ax[0], color='red')
    ax[0].title.set_text('Original Image')
    ax[0].imshow(image)

    get_bbox(aug_result['bboxes'], ax[1], color='red')
    ax[1].title.set_text('Augmented Image')
    ax[1].imshow(aug_result['image'])
    plt.show()


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


path = "data/train.csv"
train_df = pd.read_csv(path)

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

augmentation(train_df, "0a4408b37")
