from torch.utils.data import DataLoader
import torch
import dataset as ds
import model as mod
import evaluation as ev
import engine as en
from dataset import DataGenerator


def collate_fn(batch):
    return tuple(zip(*batch))


train_df, valid_df = ds.get_train_valid_data()
train_dataset = DataGenerator(train_df, ds.DIR_TRAIN, ds.get_train_transform())
valid_dataset = DataGenerator(valid_df, ds.DIR_TRAIN, ds.get_valid_transform())

train_data_loader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = mod.get_model()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(params, lr=0.002, eps=1e-08, weight_decay=5e-5)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 40
validation_accuracy = None

for epoch in range(num_epochs):
    en.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=len(train_dataset) / 12)
    val = ev.validation_score(valid_data_loader, model, device)

    if epoch % 10 == 9:
        torch.save(model.state_dict(), f'fasterrcnn_resnet_fpn{epoch}.pth')

    if validation_accuracy is None or validation_accuracy < val:
        print(f"Best validation accuracy: {val}")
        validation_accuracy = val
        torch.save(model.state_dict(), f'fasterrcnn_resnet_fpn.pth')
