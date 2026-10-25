from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# 1. Configure the website
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="centered"
)


# 2. Website heading and introduction
st.title("♻️ EcoVision AI")

st.markdown(
    """
    ### Give your waste a better destination 🌱

    Upload a photo of a waste item to discover what it is
    and learn how to dispose of it responsibly.
    """
)

st.divider()


# 3. Define the model settings
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "ecovision_resnet18_best.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 4. Load the trained model once
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# 5. Preprocess uploaded images
weights = ResNet18_Weights.DEFAULT
normalization = weights.transforms()

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=normalization.mean,
        std=normalization.std
    )
])


# 6. Step-by-step disposal guidance
DISPOSAL_GUIDANCE = {
    "cardboard": {
        "title": "How to dispose of cardboard",
        "steps": [
            "Remove plastic wrapping, tape, and other materials where practical.",
            "Flatten the cardboard box to save space.",
            "Keep the cardboard clean and dry.",
            "Place it in your local cardboard recycling collection, if accepted."
        ]
    },
    "glass": {
        "title": "How to dispose of glass",
        "steps": [
            "Identify the glass item and check your local recycling rules.",
            "Empty the container and rinse it if required.",
            "Keep broken glass separate and handle it carefully.",
            "Place acceptable glass items in the designated collection container."
        ]
    },
    "metal": {
        "title": "How to dispose of metal",
        "steps": [
            "Separate the metal item from other waste.",
            "Empty food or drink containers.",
            "Rinse cans if required by your local recycling service.",
            "Place recyclable metal in the designated recycling collection."
        ]
    },
    "paper": {
        "title": "How to dispose of paper",
        "steps": [
            "Remove plastic covers and other non-paper materials.",
            "Make sure the paper is clean and dry.",
            "Keep greasy or heavily contaminated paper separate.",
            "Place recyclable paper in the appropriate collection bin."
        ]
    },
    "plastic": {
        "title": "How to dispose of plastic",
        "steps": [
            "Check the item's recycling label, if available.",
            "Empty and rinse the container when required.",
            "Separate caps or other components if local rules ask you to.",
            "Place it in the designated recycling collection only if accepted locally."
        ]
    },
    "trash": {
        "title": "How to handle general waste",
        "steps": [
            "Check whether the item can be reused.",
            "Look for recycling options for its material.",
            "Check whether a suitable composting option exists.",
            "If no suitable option is available, place it in the appropriate general-waste bin."
        ]
    }
}


# 7. Upload a waste image
st.subheader("🔍 Identify your waste")

uploaded_file = st.file_uploader(
    "Choose an image of a waste item",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo showing the item."
)


# 8. Identify the item and show disposal steps
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

        if not MODEL_PATH.exists():
            st.error(
                "The trained model could not be found. "
                "Please check the models folder."
            )

        else:
            with st.spinner("Identifying your waste... 🌱"):

                model = load_model()

                image_tensor = eval_transform(image)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    predicted_index = outputs.argmax(dim=1).item()

                predicted_class = CLASSES[predicted_index]

            st.divider()

            st.subheader("🌿 What is this item?")

            st.success(
                f"This item appears to be: **{predicted_class.title()}**"
            )

            guidance = DISPOSAL_GUIDANCE[predicted_class]

            st.subheader(f"♻️ {guidance['title']}")

            st.write(
                "Follow these simple steps to dispose of it responsibly:"
            )

            for step_number, step in enumerate(
                guidance["steps"], start=1
            ):
                st.markdown(f"**{step_number}.** {step}")

            st.info(
                "This identification is an AI-generated suggestion "
                "and may be incorrect. Always check your local "
                "waste collection and recycling rules."
            )

    except Exception:
        st.error(
            "We couldn't process this image. "
            "Please try uploading another JPG or PNG image."
        )


# 9. Environmental quote
st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 18px 8px;
    ">
        <p style="
            font-size: 19px;
            font-style: italic;
            color: #299764;
        ">
            "Small actions today create a greener tomorrow."
        </p>
        <p style="font-size: 14px; color: gray;">
            Every item disposed of responsibly makes a difference.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("EcoVision AI · Making responsible disposal easier 🌍")