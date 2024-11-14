
from clarifai.client.model import Model
from PIL import Image
from io import BytesIO
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your PAT (Personal Access Token) can be found in the Account's Security section
# Specify the correct user_id/app_id pairings
# Since you're making inferences outside your app's scope
#USER_ID = "clarifai"
#APP_ID = "main"

# You can set the model using model URL or model ID.
# Change these to whatever model you want to use
# eg : MODEL_ID = "general-english-image-caption-blip"
# You can also set a particular model version by specifying the  version ID
# eg: MODEL_VERSION_ID = "cdb690f13e62470ea6723642044f95e4"
#  Model class objects can be inititalised by providing its URL or also by defining respective user_id, app_id and model_id

# eg : model = Model(user_id="clarifai", app_id="main", model_id=MODEL_ID)
model_url = (
    "https://clarifai.com/clarifai/main/models/food-item-v1-recognition"
)
# image_url = "https://s3.amazonaws.com/samples.clarifai.com/featured-models/image-captioning-statue-of-liberty.jpeg"

# The Predict API also accepts data through URL, Filepath & Bytes.
# Example for predict by filepath:
# model_prediction = Model(model_url).predict_by_filepath(filepath, input_type="text")

# Example for predict by bytes:
# model_prediction = Model(model_url).predict_by_bytes(image_bytes, input_type="text")
# Step 1: Upload the image

def predict_food(uploaded_file):
    # Read the uploaded file as binary
    image_bytes = uploaded_file.read()  # This reads the file in binary mode
    with st.spinner("Getting items..."):
        model_prediction = Model(url=model_url, pat="82a56de8938f4b8e899e39568b45ae34").predict_by_bytes(
            image_bytes, input_type="image"
        )
    # Get the output
    ans = model_prediction.outputs[0].data.concepts[:5]
    return [(i.name,i.value) for i in ans]





def analyze_diet(food_items, user_info):
    
    prompt = f"""You are a helpful health assistant whose job is to strictly provide information about the food or fruits given by me.
    The format of the my input will be (food_item, probability_food_item_is_present). If the probability is lower than 0.25. Ignore it.
    you can suggest healthier alternatives or things to add to make it healthier for people that fit my information?
    Food information: {str(food_items)}
    User Information:
    - Name: {user_info.name}
    - Age: {user_info.age}
    - Sex: {user_info.sex}
    - Weight: {user_info.weight}
    - Height: {user_info.height}
    - Goals: {user_info.goals}
    - Country: {user_info.country}
    """
    # You can set the model using model URL or model ID.
    model_url="https://clarifai.com/openai/chat-completion/models/gpt-4o"


    # Model Predict
    with st.spinner("Analyzing Diet..."):
        model_prediction = Model(url=model_url,pat="82a56de8938f4b8e899e39568b45ae34").predict_by_bytes(prompt.encode(), input_type="text")

    return (model_prediction.outputs[0].data.text.raw)
