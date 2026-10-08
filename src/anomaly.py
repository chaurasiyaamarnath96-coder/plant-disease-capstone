import torch
from PIL import Image
from torchvision import transforms

from autoencoder import AutoEncoder

# ==========================
# CONFIGURATION
# ==========================

DEVICE = torch.device("cpu")

MODEL_PATH = (
    r"C:\Projects\plant-disease-capstone\models\autoencoder.pth"
)

# Use your calculated threshold
THRESHOLD = 0.006064

# ==========================
# AUTOENCODER TRANSFORM
# IMPORTANT:
# Do NOT use ImageNet normalization
# ==========================

ae_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================
# LOAD MODEL
# ==========================

autoencoder = AutoEncoder()

autoencoder.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

autoencoder.to(DEVICE)
autoencoder.eval()

# ==========================
# ANOMALY DETECTOR
# ==========================

def detect_anomaly(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = ae_transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        reconstructed = autoencoder(
            image_tensor
        )

        error = torch.mean(
            (image_tensor - reconstructed) ** 2
        ).item()

    ratio = error / THRESHOLD

    return {
        "error": round(error, 6),
        "threshold": round(THRESHOLD, 6),
        "ratio": round(ratio, 2),
        "is_anomaly": error > THRESHOLD
    }


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    result = detect_anomaly(
        r"C:\Projects\plant-disease-capstone\src\7acc2f36-b465-4131-9110-265965ccdb87___RS_HL 0150.JPG"
    )

    print(result)