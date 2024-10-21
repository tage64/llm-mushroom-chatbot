#!/usr/bin/env python
# coding: utf-8

import google.generativeai as genai
print(genai.__version__)


print(genai.__version__)


import gradio as gr
import google.generativeai as genai
import os
from PIL import Image
from typing import Generator
import json




print(gr.__version__)


os.environ["API_KEY"]="AIzaSyBtne2lEYTRFDjQvXO19MsdUhfDZN_szjA"


genai.configure(api_key=os.environ["API_KEY"])


model = genai.GenerativeModel("gemini-1.5-flash")
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Complexity configurations
COMPLEXITY_CONFIGS = {
    "Low": {
        "temperature": 0.3,
        "top_k": 1,
    },
    "Medium": {
        "temperature": 0.7,
        "top_k": 3,
    },
    "High": {
        "temperature": 1.0,
        "top_k": 10,
    }
}

# Mushroom expert prompt
MUSHROOM_EXPERT_PROMPT = """
You are a mushroom expert chatbot. Your goal is to provide accurate information about mushrooms and guide the conversation towards mushroom-related topics. Answer it briefly. 
Remember, your primary focus is on mushrooms and related topics.
"""

# Safety settings
safety_settings = {
    "HATE": "BLOCK_LOW_AND_ABOVE",
    "HARASSMENT": "BLOCK_LOW_AND_ABOVE",
    "SEXUAL": "BLOCK_LOW_AND_ABOVE",
}

def process_image(image, complexity):
    if image is None:
        return None

    # First, check if the image contains a mushroom
    check_prompt = "Does this image contain a mushroom? Answer with only 'Yes' or 'No'."
    try:
        check_response = vision_model.generate_content([check_prompt, image])
        is_mushroom = check_response.text.strip().lower() == "yes."
    except Exception as e:
        print(f"An error occurred while checking the image: {str(e)}")
        return {"error": f"An error occurred while checking the image: {str(e)}"}

    if not is_mushroom:
        print("The uploaded image does not contain a mushroom.")
        return {"error": "The uploaded image does not contain a mushroom."}

    try:
        # Get JSON response
        json_prompt = """
        Provide a JSON response with the following fields:
        common_name, genus, confidence (of the prediction), visible (what parts of the mushrooms are visible in the image, from the set {cap, hymenium, stipe}), color (of the mushroom in the picture), and edible (boolean indicating if the mushroom is edible).
        
        Example format:
        {
          "common_name": "Inkcap",
          "genus": "Coprinus",
          "confidence": 0.5,
          "visible": ["cap", "hymenium", "stipe"],
          "color": "orange",
          "edible": true
        }
        
        Provide only the JSON response without any additional text.
        """
        json_response = vision_model.generate_content([json_prompt, image],
            generation_config=genai.types.GenerationConfig(**COMPLEXITY_CONFIGS[complexity]))
        json_analysis = json.loads(json_response.text)
        
        print(f"Image Analysis Result: {json.dumps(json_analysis, indent=2)}")
        return json_analysis
    except Exception as e:
        print(f"An error occurred while processing the mushroom image: {str(e)}")
        return {"error": f"An error occurred while processing the mushroom image: {str(e)}"}

def chat(message, image, complexity, history):
    history = history or []

    # Set a default message if only an image is uploaded
    if not message and image is not None:
        message = "An image has been uploaded for analysis."

    image_analysis = process_image(image, complexity) if image is not None else None

    if image_analysis:
        if "error" in image_analysis:
            full_message = f"{MUSHROOM_EXPERT_PROMPT}An error occurred during image analysis: {image_analysis['error']}. {message}"
        else:
            full_message = f"{MUSHROOM_EXPERT_PROMPT}Here's the analysis of the mushroom image: {json.dumps(image_analysis)}. "
            if message != "An image has been uploaded for analysis.":
                full_message += f"The user also asked: '{message}'. Please answer based on the image analysis and the user's question."
            else:
                full_message += "Please provide a summary of the mushroom based on this analysis."
    else:
        full_message = MUSHROOM_EXPERT_PROMPT + message if message else MUSHROOM_EXPERT_PROMPT + "Please provide a message or upload an image to start the conversation about mushrooms."

    try:
        response = model.generate_content(
            full_message,
            generation_config=genai.types.GenerationConfig(**COMPLEXITY_CONFIGS[complexity]),
            safety_settings=safety_settings,
            stream=True,
        )

        safety_triggered = False
        full_response = ""

        for chunk in response:
            if "SAFETY" in str(chunk.candidates[0].finish_reason):
                safety_triggered = True
                safety_message = "Safety filter triggered. The response may contain sensitive content."
                yield history + [[message, safety_message]]
                break

            if chunk.text:
                full_response += chunk.text
                yield history + [[message, full_response]]

        if safety_triggered:
            history.append([message, safety_message])
        else:
            history.append([message, full_response])

    except Exception as e:
        error_message = f"An error occurred: {str(e)}. Please try again or rephrase your question."
        history.append([message, error_message])
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=800)
    msg = gr.Textbox(placeholder="Type your message here (optional if image is uploaded)")
    image = gr.Image(type="pil", label="Upload a mushroom image (optional)")
    complexity_dropdown = gr.Dropdown(choices=["Low", "Medium", "High"], value="Low", label="Complexity")
    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(chat, inputs=[msg, image, complexity_dropdown, chatbot], outputs=chatbot)
    msg.submit(chat, inputs=[msg, image, complexity_dropdown, chatbot], outputs=chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()


import gradio as gr

def echo(message, history):
    return message

demo = gr.Interface(
    fn=echo,
    inputs="text",
    outputs="text",
)

demo.launch()


import importlib
importlib.reload(gr)

