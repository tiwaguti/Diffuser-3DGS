from diffuser_detector.model import DetectorUNet
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ImageDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        img_dir = os.path.join(dataset_dir, "images")
        self.filenames = [s for s in os.listdir(img_dir) if s.endswith(".png")]

    def __len__(self):
        return len(self.filenames)

    def load_data(self, idx, device="cpu"):
        filename = self.filenames[idx]
        img_path = os.path.join(self.dataset_dir, "images", filename)
        mask_path = os.path.join(self.dataset_dir, "masks", filename)
        image = np.array(Image.open(img_path))[..., :3] / 255.0
        mask = np.array(Image.open(mask_path))[..., 0:1] / 255.0 > 0.5
        return {
            "image": torch.Tensor(image).permute(2, 0, 1).float().to(device),
            "mask": torch.Tensor(mask).permute(2, 0, 1).float().to(device),
        }

    def __getitem__(self, idx):
        return self.load_data(idx)


def save_checkpoint(dst, epoch, model, optimizer):
    print("Saving checkpoint:", dst)
    device = next(model.parameters()).device
    torch.save(
        {
            "epoch": epoch,
            "model": model.cpu().state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        dst,
    )
    model.to(device)

def train(
    dataset_dir,
    output_dir,
    load_checkpoint=None,
    val_split=0.1,
    batch_size=16,
    num_epochs=1000,
    val_freq=10,
    save_freq=50,
    num_preview=10,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    torch.manual_seed(0)
    dataset = ImageDataset(dataset_dir)
    num_train = int(len(dataset) * (1 - val_split))
    train_dataset, val_dataset = random_split(
        dataset, [num_train, len(dataset) - num_train]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    model = DetectorUNet().to(device)
    optimizer = Adam(model.parameters())
    start_epoch = 0
    l1_loss = nn.L1Loss()

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    checkpoint = {"model": model, "optimizer":optimizer}

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Overall progress"):
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_dataset), desc=f"Train {epoch:5d}") as pbar:
            for data in train_loader:
                optimizer.zero_grad()
                img = data["image"].to(device)
                diff_gt = data["mask"].to(device)

                diff_pred = model(img)
                loss = l1_loss(diff_pred, diff_gt)
                loss.backward()
                optimizer.step()

                train_loss += loss
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(img.shape[0])
            
            num_data = (len(train_loader) // batch_size) * batch_size
            writer.add_scalar("loss/train_loss", train_loss / num_data, global_step=epoch)

        if epoch % val_freq == 0:
            val_loss = 0.0
            with torch.no_grad():
                model.eval()
                with tqdm(total=len(val_dataset), desc=f"  Val {epoch:5d}") as pbar:
                    for i, data in enumerate(val_loader):
                        img = data["image"].to(device)
                        diff_gt = data["mask"].to(device)
                        diff_pred = model(img)
                        loss = l1_loss(diff_pred, diff_gt)
                        val_loss += loss

                        pbar.set_postfix({"loss": loss.item()})
                        pbar.update(img.shape[0])
                        if i < num_preview and epoch == 0:
                            writer.add_image(
                                f"diffuser_{i}/input",
                                torch.clamp(img.squeeze(0), 0, 1),
                                epoch,
                            )
                            writer.add_image(
                                f"diffuser_{i}/gt",
                                torch.clamp(diff_gt.squeeze(0), 0, 1),
                                epoch,
                            )
                        if i < num_preview:
                            writer.add_image(
                                f"diffuser_{i}/pred",
                                torch.clamp(diff_pred.squeeze(0), 0, 1),
                                epoch,
                            )
            val_loss /= len(val_loader)
            writer.add_scalar("loss/val_loss", val_loss, global_step=epoch)

        if epoch % save_freq == 0:
            save_checkpoint(
                os.path.join(output_dir, f"chkpnt_{epoch:05d}.pt"),
                epoch=epoch,
                **checkpoint,
            )
    save_checkpoint(
        os.path.join(output_dir, f"chkpnt_final.pt"),
        epoch=num_epochs,
        **checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir")
    parser.add_argument("--output_dir", default="outputs/detector_train")
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()
    train(args.dataset_dir, args.output_dir, args.checkpoint_path)
