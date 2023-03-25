from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
import datetime

import sys
import gradio as gr


# Folder Name goes here
folder_name = str("midjourney")

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(folder_name):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 60

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.4, model_name="text-davinci-003", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(f"scraping/{folder_name}").load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    index.save_to_disk(f'{folder_name}_{timestamp}.json')
    # index.save_to_disk('index.json')
    return index

def chatbot(input_text):
    index_files = [file for file in os.listdir('.') if file.startswith(folder_name) and file.endswith('.json')]
    if len(index_files) == 0:
        raise ValueError(f"No index file found for the folder {folder_name}")
    elif len(index_files) > 1:
        raise ValueError(f"Multiple index files found for the folder {folder_name}")
    else:
        index = GPTSimpleVectorIndex.load_from_disk(index_files[0])
        # index = GPTSimpleVectorIndex.load_from_disk('docs/RFP/vector_files/RFP_2023-03-23-20-10-50.json')
        response = index.query(input_text, response_mode="compact")
        return response.response


def chatbot(input_text):
    # index = GPTSimpleVectorIndex.load_from_disk('docs/RFP/vector_files/RFP_2023-03-23-20-10-50.json')
    index = GPTSimpleVectorIndex.load_from_disk('midjourney_2023-03-25-00-29-45.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot, inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text", title="Midjourney 5 Language Model")

iface.launch(share=True)


# index = construct_index("docs")
index = construct_index('midjourney')

