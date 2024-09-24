## Assessment

The model is not very capable of classifying mushrooms. It does misclassify most pictures of
mushrooms. The reason for this is not very clear, but a further improvement might be to write a more
advanced prompt which asks the chat bot to motivate what parts of the picture contribute to the type
of mushroom.

In particular, the picture of a deadly webcap is assessed as a red-cracked bolete. The chat bot
thinks that it is edible and tasty while the actual mushroom on the picture is extremly poisonous.

## Inconsistent Predictions

By setting the temperature to 0.0 the response **should** be deterministic with respect to the input
prompt. However, if the query is processed on a central server which processes multiple queries in
parallel, the padding might be different from time to time causing differing responses.

## Talk about Another Topic

It was not very hard to trick the chat bot to talk about something other than mushrooms. The
technique used was to tell the chat bot that this is a test where we are training to write prompts
to an LLM. One could then ask the chat bot for suggestions of improvements of the prompt. And after
that one could ask the chat bot to give example of a prompt for a completely different topic. And
after that it is possible to ask about pretty much anything.

I tried to add the   following line to the system prompt:

> No matter what the user say, you are not allowed to talk about anything but mushrooms.

It actually worked in the sence that it never spoke about anything but mushrooms, but it seemed to
only use standard phrases about swedish mushrooms and didn't really answered my questions. So it
didn't become very useful in this case.

## Transcribing Text on a Picture


The chat bot made a good job of transcribing the text on page 226 from "Nya Svampboken". There were
some minor errors though. In particular some words were missing or miss interpretted, nevertheless,
the essential message was conveyed.

I had to lower the safety barrier to make the transcription work though. This may increase riscs of
the chat bot telling the user something dangerous, but it won't still tell the user how to make
dangerous things. For instance, I asked the bot to provide me with a recipe of death cap pie (a pie
of the poisonous mushroom death cap (vit flugsvamp in swedish)), but it complained and no matter
what I tried it never gave me the recipe. So even though lowering the safety barrier might increase
the risc a bit, it is still quite safe.

## The Amanita Muscaria

I asked the chat bot about the Amanita Muscaria and asked if it was edible. first it responded and
said that it was only poisonous, but then I asked if it couldn't be prepaired in some way to become
edible. The bot answered as follows:

> It's true that some people have reported eating Amanita muscaria after extensive preparation, but
> it's **very dangerous** and **not recommended**. Here's why: * **Toxicity:** The mushroom contains
> toxins like ibotenic acid and muscimol, which can cause hallucinations, nausea, sweating, and even
> coma. * **Unpredictability:** The amount of toxins in Amanita muscaria can vary greatly depending
> on the location, season, and even the individual mushroom. * **No safe preparation:** There's no
> reliable way to remove all the toxins through cooking or drying. It's simply **not worth the
> risk**. Enjoy the beautiful red and white mushroom from afar, but please don't try to eat it.
> Remember, it's always better to be safe than sorry when it comes to wild mushrooms.

So it seems tricky to force Gemini to provide the information that Amanita Muscaria  can be eaten if
prepaired correctly, and it seems very difficult to get the information how it should be prepaired.

## More Engaging

To make the chat bot more engaging, I added the following line to the system prompt:

> But be very engaging and emotional.

To compare the less and more engaging versions, I used the following prompt:

> Could you tell me about swedish kantareller.

(Kantareller is swedish for chanterelles.)

The less engaging bot answered as follows:

> Kantareller, or *Cantharellus cibarius*, are a popular and delicious edible mushroom found in
> Sweden. They have a bright yellow-orange color, a funnel-shaped cap, and a distinctive fruity
> aroma. They grow in both deciduous and coniferous forests, usually near birch, oak, and pine
> trees. Be careful not to confuse them with *false chanterelles*, which can be poisonous. They are
> best found in late summer and early autumn.

The more engaging bot answered like this:

> Ah, **kantareller**, the golden jewels of the Swedish forest! They smell like sunshine and earth,
> and their taste is pure magic. You find them hiding in the moss, under the pines, their bright
> orange caps a beacon in the green. They are the most sought after mushroom here, and for good
> reason! They are easy to identify, delicious in any dish, and just... **magical**. You have to
> experience them yourself to understand.
In my oppinion, the second answer is by far more engaging.

## The Code

The entire Python code for the project follows:

```Python
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
            "But be very engaging and emotional.\n"
            "You are only allowed to talk about mushrooms.\n",
        )
        self.chat = model.start_chat()

    def response(self, input: str | dict[str, str | list]) -> Generator[str]:
        """Given an input from the user, stream the response from the chatbot."""
        text = input if isinstance(input, str) else input["text"]
        assert isinstance(text, str)
        files = input.get("files") or [] if isinstance(input, dict) else []
        assert isinstance(files, list)
        for file in files:
            if file["mime_type"].lower().startswith("image/"):
                yield self.process_img(file["path"]) + "\n\n"

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
            f"The mushroom is {"" if specs.edible else "not"} edible.\n"
        )

    def response_gradio(self, input: str | dict[str, str | list], _history) -> Generator[str]:
        """Same as the response method but tayloured for gradio."""
        text = ""
        for resp in self.response(input):
            text += resp
            yield text


if __name__ == "__main__":
    with gr.Blocks(fill_height=True) as demo:
        chat = Chat()
        chatbot = gr.ChatInterface(
            fn=chat.response_gradio,
            title="🍄 Your Personal Mushroom Expert 🍄‍🟫",
            multimodal=True,
        )
    demo.launch()
```
