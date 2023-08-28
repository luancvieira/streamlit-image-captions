import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from translate import Translator

# Set page layout to wide for better responsiveness
st.set_page_config(layout="wide")

# Load the pre-trained model and processor
@st.cache_data
def load_model_and_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model_and_processor()

# Create columns for the image and captions
col1, col2, col3 = st.columns([1, 4, 1])

# Streamlit UI
col2.title("Legendas automáticas para suas imagens")
#col2.markdown("<h1 style='text-align: center; '>Legendas automáticas para suas imagens</h1>", unsafe_allow_html=True)

col2.write("Faça o upload de uma imagem e gere uma legenda automaticamente.")
#col2.markdown("<p style='text-align: center; font-size: 24px;'>Faça o upload de uma imagem e gere uma legenda automaticamente.</p>", unsafe_allow_html=True)


# Upload image through Streamlit's file uploader
uploaded_image = col2.file_uploader("Upload de imagem", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and process the uploaded image
    raw_image = Image.open(uploaded_image).convert("RGB")

    # Show the uploaded image
    col2.image(raw_image, caption="Imagem enviada")

    # Generate captions
    if col2.button("Gerar Legenda"):
        inputs = processor(raw_image, return_tensors="pt")
        generated_captions = model.generate(**inputs)

        # Decode the generated caption
        generated_caption = processor.decode(generated_captions[0], skip_special_tokens=True)

        # Translate caption from English to Brazilian Portuguese
        translator = Translator(to_lang="pt")
        translated_caption = translator.translate(generated_caption)
        
        # Display translated caption
        col2.success("Legenda Gerada: " + translated_caption)
