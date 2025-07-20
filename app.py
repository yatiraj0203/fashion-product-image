import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MultiOutputModel
import pickle

# Load model
model = MultiOutputModel(47, 46, 6, 3)  # Replace with actual class counts
model.load_state_dict(torch.load("fashion_model.pth", map_location="cpu"))
model.eval()

# Load encoders
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

st.title("👕 Fashion Product Classifier")

file = st.file_uploader("Upload a fashion product image (.jpg)", type=["jpg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        out1, out2, out3, out4 = model(input_tensor)
        pred = {
            "Article Type": encoders['articleType'].inverse_transform([torch.argmax(out1).item()])[0],
            "Base Colour": encoders['baseColour'].inverse_transform([torch.argmax(out2).item()])[0],
            "Season": encoders['season'].inverse_transform([torch.argmax(out3).item()])[0],
            "Gender": encoders['gender'].inverse_transform([torch.argmax(out4).item()])[0]
        }

    st.markdown("### 🎯 Prediction")
    for key, val in pred.items():
        st.write(f"**{key}:** {val}")
