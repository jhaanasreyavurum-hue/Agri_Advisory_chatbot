import os
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------
# GOOGLE DRIVE DIRECT LINKS
# Replace with your real file IDs
# -------------------------
maize_url = "https://drive.google.com/uc?export=download&id=1BXdS0khtYvGhAVLi9aqfI9KOILYRhorj"
disease_url = "https://drive.google.com/uc?export=download&id=1SSwyapIc8wYIVn4UiVaaJ7_zJc3g0JXf"
pest_url = "https://drive.google.com/uc?export=download&id=1QipEeKTWcVG9lol7dV32WZMFHRowDl_i"

# -------------------------
# CLASS LABELS
# -------------------------
maize_classes = ["maize", "not_maize"]
disease_classes = ["blight", "grey_leaf_spot", "healthy", "rust"]
pest_classes = ["aphids", "corn_rootworm", "fall_army_worm", "stalk_borer"]

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# DOWNLOAD FUNCTION
# -------------------------
def download_file(url, path):
    if not os.path.exists(path):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        print(f"Downloading {path} from {url} ...")
        urllib.request.urlretrieve(url, path)

# -------------------------
# LOAD MODELS
# -------------------------
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model files if missing
    download_file(maize_url, "models/maize_check_model.pth")
    download_file(disease_url, "models/maize_disease_model.pth")
    download_file(pest_url, "models/maize_pest_model.pth")

    # Maize / not_maize model
    maize_model = models.resnet18(weights=None)
    maize_model.fc = nn.Linear(maize_model.fc.in_features, 2)
    maize_model.load_state_dict(torch.load("models/maize_check_model.pth", map_location=device))
    maize_model.to(device)
    maize_model.eval()

    # Disease model
    disease_model = models.resnet18(weights=None)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, 4)
    disease_model.load_state_dict(torch.load("models/maize_disease_model.pth", map_location=device))
    disease_model.to(device)
    disease_model.eval()

    # Pest model
    pest_model = models.resnet18(weights=None)
    pest_model.fc = nn.Linear(pest_model.fc.in_features, 4)
    pest_model.load_state_dict(torch.load("models/maize_pest_model.pth", map_location=device))
    pest_model.to(device)
    pest_model.eval()

    return maize_model, disease_model, pest_model

# -------------------------
# PREDICT IMAGE
# -------------------------
def predict_image(image, maize_model, disease_model, pest_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: maize check
        maize_output = maize_model(img_tensor)
        maize_probs = torch.softmax(maize_output, dim=1)
        maize_conf, maize_pred = torch.max(maize_probs, dim=1)

        maize_label = maize_classes[maize_pred.item()]
        maize_confidence = round(maize_conf.item() * 100, 2)

        if maize_label == "not_maize":
            return "not_maize", maize_confidence, "maize_check"

        # Step 2: disease prediction
        disease_output = disease_model(img_tensor)
        disease_probs = torch.softmax(disease_output, dim=1)
        disease_conf, disease_pred = torch.max(disease_probs, dim=1)

        disease_label = disease_classes[disease_pred.item()]
        disease_confidence = round(disease_conf.item() * 100, 2)

        # Step 3: pest prediction
        pest_output = pest_model(img_tensor)
        pest_probs = torch.softmax(pest_output, dim=1)
        pest_conf, pest_pred = torch.max(pest_probs, dim=1)

        pest_label = pest_classes[pest_pred.item()]
        pest_confidence = round(pest_conf.item() * 100, 2)

        # Step 4: choose higher confidence
        if disease_confidence >= pest_confidence:
            return disease_label, disease_confidence, "disease"
        else:
            return pest_label, pest_confidence, "pest"
