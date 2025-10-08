import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import gdown, zipfile

# ✅ NEW (server dataset location)
SERVER_DATA_DIR = os.path.abspath("./server_data")
SERVER_IMG_DIR = os.path.join(SERVER_DATA_DIR, "img")
SERVER_CSV_PATH = os.path.join(SERVER_DATA_DIR, "server_test.csv")
SERVER_ZIP_PATH = os.path.join(SERVER_DATA_DIR, "server_data.zip")

# 🔑 Google Drive link for auto-download
GDRIVE_FILE_ID = "1PEjxFajumhAopGlLxpirXmH5jJ_FQ3AW"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# ✅ Keep client paths unchanged
BASE_DATASET_DIR = os.path.abspath("../../dataset")
IMG_DIR = os.path.abspath("../../all_in_one_dataset/img")


def load_partition_for_client(client_id: int):
    """Load train/val/test CSVs for a given client."""
    client_dir = os.path.join(BASE_DATASET_DIR, f"client_{client_id}")
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(client_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"))
    return train_df, val_df, test_df


def load_server_test_data(limit: int = 200):
    """Load server_test.csv (download & unzip if missing).
       Optionally limit number of samples to reduce CPU usage."""
    if not os.path.exists(SERVER_CSV_PATH):
        print(f"⚠️ server_test.csv not found at {SERVER_CSV_PATH}")
        print("⬇️ Downloading from Google Drive...")

        os.makedirs(SERVER_DATA_DIR, exist_ok=True)

        # Download zip
        gdown.download(GDRIVE_URL, SERVER_ZIP_PATH, quiet=False)

        # Unzip contents into server_data/
        with zipfile.ZipFile(SERVER_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(SERVER_DATA_DIR)

        if not os.path.exists(SERVER_CSV_PATH):
            raise FileNotFoundError(
                f"❌ server_test.csv still not found after download/unzip at {SERVER_CSV_PATH}"
            )

    df = pd.read_csv(SERVER_CSV_PATH)

    # ✅ Subsample to avoid CPU overload
    if limit and len(df) > limit:
        print(f"⚠️ Limiting server_test.csv from {len(df)} → {limit} samples")
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    return df


def gl_model_torch_validation(batch_size: int = 32, max_len: int = 128):
    """Return DataLoader for global server validation (server_test.csv)."""
    df = load_server_test_data(limit=200)
    dataset = HatefulMemesDataset(df, max_len=max_len, use_server_data=True)
    # ✅ Use num_workers=0 to avoid CPU spike in Kubernetes
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class HatefulMemesDataset(Dataset):
    def __init__(self, df, max_len=128, use_server_data=False):
        print(f"🗂️ Creating HatefulMemesDataset with {len(df)} samples...")
        self.df = df.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.max_len = max_len

        # ✅ Use server images if required
        if use_server_data:
            if not os.path.exists(SERVER_IMG_DIR):
                raise FileNotFoundError(
                    f"❌ Server images directory not found: {SERVER_IMG_DIR}"
                )
            self.cache_dir = SERVER_IMG_DIR
        else:
            if not os.path.exists(IMG_DIR):
                raise FileNotFoundError(
                    f"❌ Local image directory not found: {IMG_DIR}\n"
                    "Make sure you ran `git lfs pull` or have local data."
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
