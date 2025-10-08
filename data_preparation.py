import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image

# ✅ OLD (local full dataset)
# BASE_DATASET_DIR = os.path.abspath("../../dataset")
# IMG_DIR = os.path.abspath("../../all_in_one_dataset/img")

# ✅ NEW (only server_data folder from Drive)
SERVER_DATA_DIR = os.path.abspath("./server_data")  # after download
SERVER_IMG_DIR = os.path.join(SERVER_DATA_DIR, "img")
SERVER_CSV_PATH = os.path.join(SERVER_DATA_DIR, "server_test.csv")

# ✅ Keep client paths unchanged (you still have local data for clients)
BASE_DATASET_DIR = os.path.abspath("../../dataset")
IMG_DIR = os.path.abspath("../../all_in_one_dataset/img")


def load_partition_for_client(client_id: int):
    """Load train/val/test CSVs for a given client."""
    client_dir = os.path.join(BASE_DATASET_DIR, f"client_{client_id}")
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(client_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"))
    return train_df, val_df, test_df


def load_server_test_data():
    """Load the server_test.csv from server_data/."""
    if not os.path.exists(SERVER_CSV_PATH):
        raise FileNotFoundError(
            f"❌ server_test.csv not found at: {SERVER_CSV_PATH}\n"
            "Make sure the Drive download/unzip step happens before loading data."
        )
    return pd.read_csv(SERVER_CSV_PATH)
from torch.utils.data import DataLoader

def gl_model_torch_validation(batch_size: int = 32, max_len: int = 128):
    """Return DataLoader for global server validation (using server_test.csv)."""
    df = load_server_test_data()
    dataset = HatefulMemesDataset(df, max_len=max_len, use_server_data=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class HatefulMemesDataset(Dataset):
    def __init__(self, df, max_len=128, use_server_data=False):
        """
        use_server_data=True → load images from server_data/img/
        otherwise → use client/local image dir.
        """
        print(f"🗂️ Creating HatefulMemesDataset with {len(df)} samples...")
        self.df = df.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.max_len = max_len

        # ✅ Decide which image folder to use
        if use_server_data:
            if not os.path.exists(SERVER_IMG_DIR):
                raise FileNotFoundError(
                    f"❌ Server images directory not found: {SERVER_IMG_DIR}\n"
                    "Ensure the folder is downloaded first."
                )
            self.cache_dir = SERVER_IMG_DIR
        else:
            if not os.path.exists(IMG_DIR):
                raise FileNotFoundError(
                    f"❌ Local image directory not found: {IMG_DIR}\n"
                    f"Make sure you ran `git lfs pull` or have local data."
                )
            self.cache_dir = IMG_DIR

        print(f"📁 Using image directory: {self.cache_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize text
        tok = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # Load image
        img_name = os.path.basename(row["img_name"])
        img_path = os.path.join(self.cache_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            image = self.transform(img)
        except Exception as e:
            print(f"⚠️ Failed to load image at {img_path}: {e}")
            image = torch.zeros((3, 224, 224))

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }
