from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import MultiOutputModel  # Ensure this is in your project
import io
import pickle

app = FastAPI()

# Load model and label maps
model = MultiOutputModel(47, 46, 6, 3)  # Use exact class counts
model.load_state_dict(torch.load("fashion_model.pth", map_location="cpu"))
model.eval()

with open("label_encoders.pkl", "rb") as f:
    label_maps = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        out1, out2, out3, out4 = model(image)
        preds = {
            "articleType": label_maps['articleType'].inverse_transform([torch.argmax(out1).item()])[0],
            "baseColour": label_maps['baseColour'].inverse_transform([torch.argmax(out2).item()])[0],
            "season": label_maps['season'].inverse_transform([torch.argmax(out3).item()])[0],
            "gender": label_maps['gender'].inverse_transform([torch.argmax(out4).item()])[0]
        }
    return preds
