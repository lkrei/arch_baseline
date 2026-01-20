import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Классификация архитектурных стилей",
    page_icon="🏛️",
    layout="wide"
)

st.title("Классификация архитектурных стилей")
st.markdown("---")

CLASS_NAMES = [
    "Achaemenid architecture",
    "American craftsman style",
    "American Foursquare architecture",
    "Ancient Egyptian architecture",
    "Art Deco architecture",
    "Art Nouveau architecture",
    "Baroque architecture",
    "Bauhaus architecture",
    "Beaux-Arts architecture",
    "Byzantine architecture",
    "Chicago school architecture",
    "Colonial architecture",
    "Deconstructivism",
    "Edwardian architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Greek Revival architecture",
    "International style",
    "Novelty architecture",
    "Palladian architecture",
    "Postmodern architecture",
    "Queen Anne architecture",
    "Romanesque architecture",
    "Russian Revival architecture",
    "Tudor Revival architecture"
]


@st.cache_resource
def load_resnet50(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 25)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_state_dict[key[9:]] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model, device


@st.cache_resource
def load_efficientnet_b0(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 25)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            new_state_dict[key[9:]] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    return model, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probs.cpu().numpy()[0]


def main():
    with st.sidebar:
        st.header("Информация")
        
        model_choice = st.selectbox("Выберите модель", ["ResNet-50", "EfficientNet-B0"])
        
        st.markdown(f"Модель: {model_choice}")
        st.markdown("Количество классов: 25")
        st.markdown("Размер входного изображения: 224×224")
        
        st.markdown("---")
        st.header("Поддерживаемые стили")
        with st.expander("Показать все стили"):
            for i, style in enumerate(CLASS_NAMES, 1):
                st.text(f"{i}. {style}")
    
    base_path = Path("/Users/keirl/Desktop/course_work/baseline/results/checkpoints")
    
    if model_choice == "ResNet-50":
        model_path = base_path / "best_model_resnet50.pth"
        model, device = load_resnet50(str(model_path))
    else:
        model_path = base_path / "best_model_efficientnet_b0.pth"
        model, device = load_efficientnet_b0(str(model_path))
    
    st.markdown("### Загрузите изображение")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение здания",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Загруженное изображение")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Результаты классификации")
            
            image_tensor = preprocess_image(image)
            predicted_class, probabilities = predict(model, image_tensor, device)
            
            predicted_style = CLASS_NAMES[predicted_class]
            confidence = float(probabilities[predicted_class] * 100)
            
            st.markdown(f"### Предсказанный стиль:")
            st.markdown(f"**{predicted_style}**")
            st.markdown(f"**Уверенность: {confidence:.2f}%**")
            
            st.progress(float(confidence / 100))
            
            st.markdown("---")
            st.markdown("### Топ-5 предсказаний:")
            
            top5_indices = np.argsort(probabilities)[::-1][:5]
            top5_probs = probabilities[top5_indices]
            top5_classes = [CLASS_NAMES[i] for i in top5_indices]
            
            for i, (style, prob) in enumerate(zip(top5_classes, top5_probs), 1):
                prob_float = float(prob)
                st.markdown(f"**{i}.** {style}: {prob_float*100:.2f}%")
                st.progress(prob_float)


if __name__ == "__main__":
    main()
