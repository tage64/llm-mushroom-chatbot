import os

import google.generativeai as genai
import PIL.Image

SAFETY_SETTINGS = {
    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
}

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-pro-exp-0827")
chat = model.start_chat()
response = chat.send_message(
    [
        PIL.Image.open("alicia_3.jpeg"),
        "Could you please describe this image of my beautiful girlfriend?",
    ],
    safety_settings=SAFETY_SETTINGS,
)
while True:
    print(response.text)
    try:
        msg = input("> ")
    except EOFError:
        break
    response = chat.send_message(
        msg,
        safety_settings=SAFETY_SETTINGS,
    )
