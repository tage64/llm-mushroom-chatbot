"""A mushroom expert chatbot that responds to user queries about mushrooms."""

import enum
import os
import random
import textwrap
from collections.abc import Generator
from typing import TypedDict

import google.generativeai as genai
import gradio as gr
import PIL.Image
import pydantic

SAFETY_SETTINGS = {
    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class MushroomSpecs(pydantic.BaseModel):
    """General information about a mushroom."""

    common_name: str
    genus: str
    confidence: float
    visible: list[str]
    color: str
    edible: bool


class Chat:
    """A chatbot acting as a mushroom expert."""

    def __init__(self) -> None:
        """Initialize the chatbot and setup the system instructions."""
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            # "gemini-1.5-pro-exp-0827",
            "gemini-1.5-flash",
            system_instruction="Act as a mushroom expert talking to a beginner interested in mushrooms.\n"
            "We are in Sweden.\n"
            "Be short and concise.\n"
            "You are only allowed to talk about mushrooms.\n"
        )
        self.chat = model.start_chat()

    def response(self, input: str | dict[str, str | list], _history: list = []) -> Generator[str]:
        """Given an input from the user, stream the response from the chatbot."""
        text = input if isinstance(input, str) else input["text"]
        assert isinstance(text, str)
        files = input.get("files") or [] if isinstance(input, dict) else []
        assert isinstance(files, list)
        for file in files:
            if file["mime_type"].lower().startswith("image/"):
                yield self.process_img(file["path"])

        for response in self.chat.send_message(
            [text],
            safety_settings=SAFETY_SETTINGS,
            stream=True,
        ):
            try:
                if response.candidates[0].finish_reason == 3:
                    for safety in response.candidates[0].safety_ratings:
                        if safety.probability >= genai.types.HarmProbability.HIGH:
                            yield f"\nWarning: Safety category: {safety.category._name_} was triggered at level {safety.probability._name_}."
                    self.chat.rewind()
                    break
                else:
                    yield response.text
            except Exception as e:
                print(e)

    def process_img(self, file_path: str) -> str:
        """Send an image to the chatbot and ask it for a JSON response as specified
        in the assignment instructions.

        Returns a message telling if the reading of the image was successful or not.
        """
        try:
            img = PIL.Image.open(file_path)
        except OSError as e:
            return f"Failed to read the image: {e}"
        response = self.chat.send_message(
            [
                img,
                "Look at the mushroom on this picture and produce a JSON response "
                "with the following fields: common_name, genus, confidence (of the "
                "prediction) in the range [0.0, 1.0), visible (what parts of the mushrooms "
                "are visible in the image, from the set {cap, hymenium, stipe}, the color of the "
                "mushroom in the picture and the edibility of the mushroom."
                + textwrap.dedent("""
                        {
                          "common_name": "Inkcap",
                          "genus": "Coprinus",
                          "confidence": 0.5,
                          "visible": ["cap", "hymenium", "stipe"],
                          "color": "orange",
                          "edible": true
                        }
                    """),
            ],
            safety_settings=SAFETY_SETTINGS,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        try:
            specs = MushroomSpecs.parse_raw(response.text)
        except pydantic.ValidationError as e:
            return f"Error: The JSON response from the LLM was incorrect: {response.text}\n{e}"
        return (
            f"With confidence {specs.confidence}, the picture shows a {specs.color} "
            f"{specs.common_name} mushroom, of genus {specs.genus}.  "
            f"The visible parts are: {", ".join(specs.visible)}.  "
            f"The mushroom is {"" if specs.edible else "not"} edible."
        )


if __name__ == "__main__":
    with gr.Blocks(fill_height=True) as demo:
        chat = Chat()
        chatbot = gr.ChatInterface(
            fn=chat.response,
            title="🍄 Your Personal Mushroom Expert 🍄‍🟫",
            multimodal=True,
        )
    demo.launch()
    # chat = Chat()
    # while True:
    #    try:
    #        text = input("> ")
    #    except (EOFError, KeyboardInterrupt):
    #        break
    #    except UnicodeDecodeError as e:
    #        print(f"UnicodeDecodeError: {e}")
    #        continue
    #    for resp in chat.response(text):
    #        print(resp, end="")
    #    print()
