import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "jhaanasreya/maize-advisory-models"

maize_classes = ["maize", "not_maize"]
disease_classes = ["blight", "grey_leaf_spot", "healthy", "rust"]
pest_classes = ["aphids", "corn_rootworm", "fall_army_worm", "stalk_borer"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Downloading models from Hugging Face...")

    maize_path = hf_hub_download(repo_id=REPO_ID, filename="maize_check_model.pth")
    disease_path = hf_hub_download(repo_id=REPO_ID, filename="maize_disease_model.pth")
    pest_path = hf_hub_download(repo_id=REPO_ID, filename="maize_pest_model.pth")

    print("All models downloaded. Loading...")

    maize_model = models.resnet18(weights=None)
    maize_model.fc = nn.Linear(maize_model.fc.in_features, 2)
    maize_model.load_state_dict(torch.load(maize_path, map_location=device, weights_only=True))
    maize_model.to(device)
    maize_model.eval()

    disease_model = models.resnet18(weights=None)
    disease_model.fc = nn.Linear(disease_model.fc.in_features, 4)
    disease_model.load_state_dict(torch.load(disease_path, map_location=device, weights_only=True))
    disease_model.to(device)
    disease_model.eval()

    pest_model = models.resnet18(weights=None)
    pest_model.fc = nn.Linear(pest_model.fc.in_features, 4)
    pest_model.load_state_dict(torch.load(pest_path, map_location=device, weights_only=True))
    pest_model.to(device)
    pest_model.eval()

    print("All models loaded successfully!")
    return maize_model, disease_model, pest_model


def predict_image(image, maize_model, disease_model, pest_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        maize_output = maize_model(img_tensor)
        maize_probs = torch.softmax(maize_output, dim=1)
        maize_conf, maize_pred = torch.max(maize_probs, dim=1)

        maize_label = maize_classes[maize_pred.item()]
        maize_confidence = round(maize_conf.item() * 100, 2)

        if maize_label == "not_maize" or maize_confidence < 90:
            return "not_maize", maize_confidence, "maize_check"

        disease_output = disease_model(img_tensor)
        disease_probs = torch.softmax(disease_output, dim=1)
        disease_conf, disease_pred = torch.max(disease_probs, dim=1)

        disease_label = disease_classes[disease_pred.item()]
        disease_confidence = round(disease_conf.item() * 100, 2)

        pest_output = pest_model(img_tensor)
        pest_probs = torch.softmax(pest_output, dim=1)
        pest_conf, pest_pred = torch.max(pest_probs, dim=1)

        pest_label = pest_classes[pest_pred.item()]
        pest_confidence = round(pest_conf.item() * 100, 2)

        if disease_confidence >= pest_confidence:
            return disease_label, disease_confidence, "disease"
        else:
            return pest_label, pest_confidence, "pest"
