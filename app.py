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

paraphrase_compiled = None

# Define the Gradio app
def paraphrase_text(text, api_key):
    global paraphrase_compiled
    if paraphrase_compiled is None:
        # Configure the LM with the provided API key
        llama = dspy.GROQ(model='mixtral-8x7b-32768', api_key=api_key, max_tokens=16384)
        dspy.settings.configure(lm=llama)

        # Compile the paraphrasing module
        train_examples = [
            ("William Shakespeare was born in Stratford-on-Avon in April (probably April 23), 1564. His father was a citizen of some prominence who became an alderman and bailiff, but who later suffered financial reverses. Shakespeare presumably attended the Stratford grammar school, where he could have acquired a respectable knowledge of Latin, but he did not proceed to Oxford or Cambridge. There are legends about Shakespeare's youth but no documented facts.",
            "William Shakespeare was born in 1564 in Stratford-on-Avon. His father, a respected alderman, and bailiff was an affluent community member but later lost his financial security. Experts suspect that Shakespeare went to the Stratford grammar school where he probably obtained a command of the Latin language, however, since there are no documented facts about his childhood, scholars rely on rumors and stories believed to be historically accurate. They do know that he did not continue his education at Oxford or Cambridge."),
            ("Exercise can help a lot in alleviating stress - that is a known fact. Exercise is a good way of reducing stress, and cardiovascular exercise is recommended for about 15 to 30 minutes, thrice or four times a week. Several studies have indicated the effects of exercise in handling stress. The activity can release endorphins to the bloodstream.",
            "Exercise is a good way to get rid of stress. It is also a perfect way of lowering stress levels, while cardiovascular exercises that can be done three or four times a week for about 15 to 30 minutes is highly suggested. Research shows that there are positive effects of exercise in dealing with stress. It can help in releasing endorphins in the body."),
            ("Dogs can provide great assistance to both children and elderly people in their daily activities. Since dogs are active pets, they can also prove to be the perfect buddies during exercise. There are different types of dogs that you can choose from to make as pets.",
            "Dogs offer help to the elderly and children in their daily life. Dogs are active pets, providing to be great exercise buddies. There are various breeds of dogs that you can select from for your pets."),
        ]
        train = [dspy.Example(text=text, answer=paraphrased_text).with_inputs('text') for text, paraphrased_text in train_examples]
        metric_EM = dspy.evaluate.answer_exact_match
        teleprompter = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2)
        paraphrase_compiled = teleprompter.compile(ParaphraseModule(), trainset=train)

    # Generate the paraphrased text
    paraphrased_text = paraphrase_compiled(text)
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
