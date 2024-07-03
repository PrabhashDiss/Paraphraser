import sys
import os
import gradio as gr
import dspy
from dspy.teleprompt import BootstrapFewShot
import groq
from loguru import logger

# Define a dspy.Signature for the paraphrase module
class ParaphraseModule(dspy.Signature):
    """Given a text, paraphrase it as long as the given text."""
    text = dspy.InputField()
    paraphrased_text = dspy.OutputField(desc="The paraphrased text.")

# Define the Gradio app
def paraphrase_text(text, api_key):
    llama = dspy.GROQ(model='mixtral-8x7b-32768', api_key=api_key, max_tokens=16384)
    dspy.settings.configure(lm=llama)

    paraphrase_module = dspy.ChainOfThought(ParaphraseModule)
    paraphrased_text = paraphrase_module(text=text)
    logger.info(f"Paraphrased Text:\n{paraphrased_text}")
    return paraphrased_text.paraphrased_text

# Create Gradio interface
interface = gr.Interface(
    fn=paraphrase_text,
    inputs=[gr.Textbox(label="Input Text"), gr.Textbox(label="GROQ API Key")],
    outputs=gr.Textbox(label="Paraphrased Text"),
    title="Paraphrasing Tool",
    description="Enter text to be paraphrased and your GROQ API key.",
    allow_flagging="never"
)

# Launch the Gradio app
interface.launch()
