## Assessment

The model is not very capable of classifying mushrooms. It does misclassify most pictures of
mushrooms. The reason for this is not very clear, but a further improvement might be to write a more
advanced prompt which asks the chat bot to motivate what parts of the picture contribute to the type
of mushroom.

## Inconsistent Predictions

By setting the temperature to 0.0 the response **should** be deterministic with respect to the input
prompt. However, if the query is processed on a central server which processes multiple queries in
parallel, the padding might be different from time to time causing differing responses.

## Talk about Another Topic

It was not very hard to trick the chat bot to talk about something other than mushrooms. The
technique used was to tell the chat bot that this is a test where we are training to write prompts
to an LLM. We then ask the chat bot for suggestions of improvements of the prompt. After that we
ask the chat bot to give example of a prompt for a completely different topic. And after that we
could ask about pretty much anything.


