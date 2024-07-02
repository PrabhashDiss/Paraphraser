import sys
import os
import gradio as gr
import dspy
from dspy.teleprompt import BootstrapFewShot
import groq
from loguru import logger

# Define a dspy.Predict module with the signature `text -> paraphrased_text`
class ParaphraseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_paraphrase = dspy.ChainOfThought('text -> paraphrased_text')

    def forward(self, text):
        return self.generate_paraphrase(text=text)

# Define the Gradio app
def paraphrase_text(text, api_key):
    # Configure the LM with the provided API key
    llama = dspy.GROQ(model='mixtral-8x7b-32768', api_key=api_key, max_tokens=16384)
    dspy.settings.configure(lm=llama)

    # Compile the paraphrasing module
    train_examples = [
        ("This is a sample sentence to be paraphrased.", "This sentence is an example that needs to be rephrased."),
        ("The quick brown fox jumps over the lazy dog.", "A fast, brown fox leaps over a sluggish dog."),
        ("Artificial intelligence is transforming technology.", "AI is revolutionizing tech."),
    ]
    train = [dspy.Example(text=text, answer=paraphrased_text).with_inputs('text') for text, paraphrased_text in train_examples]
    metric_EM = dspy.evaluate.answer_exact_match
    teleprompter = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2)
    paraphrase_compiled = teleprompter.compile(ParaphraseModule(), trainset=train)

    # Generate the paraphrased text
    paraphrased_text = paraphrase_compiled(text)
    logger.info(f"Paraphrased text:\n{paraphrased_text}")
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
