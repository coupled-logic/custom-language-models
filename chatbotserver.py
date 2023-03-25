from gpt_index import GPTSimpleVectorIndex
import gradio as gr
import os

# Folder Name goes here
folder_name = str("RFP")

# Load the index from disk
index_files = [file for file in os.listdir('.') if file.startswith(folder_name) and file.endswith('.json')]
if len(index_files) == 0:
    raise ValueError(f"No index file found for the folder {folder_name}")
elif len(index_files) > 1:
    raise ValueError(f"Multiple index files found for the folder {folder_name}")
    # get the index of the most recent file
else:
    index = GPTSimpleVectorIndex.load_from_disk(index_files[0])


def chatbot(input_text):
    response = index.query(input_text, response_mode="compact")
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title=f"Now querying {folder_name} model")

iface.launch(share=True)
