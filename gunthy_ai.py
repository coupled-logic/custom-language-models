from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
import datetime
import gradio as gr
import logging
logging.basicConfig(filename="/gunthy_ai.log", level=logging.INFO)
# Folder Name goes here
folder_name = 'gunthytrainingdata'
os.environ["OPENAI_API_KEY"] = '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'

def construct_index(folder_name):
    max_input_size = 2000
    num_outputs = 500
    max_chunk_overlap = 100
    chunk_size_limit = 500

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.4, model_name="text-davinci-003", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(folder_name).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    index_files = [file for file in os.listdir(folder_name) if
    file.startswith('gunthy_model_001') and file.endswith('.json')]
    index_file_path = os.path.join(folder_name, f'gunthy_model_001_{timestamp}.json')

    index.save_to_disk(index_file_path)
    logging.info(f"Index file saved at: {index_file_path}")
    return index

def chatbot(folder_name, input_text):
    index_files = [file for file in os.listdir(folder_name) if
                   file.startswith('gunthy_model_001') and file.endswith('.json')]
    # In chatbot function
    logging.info(f"Files in folder {folder_name}: {os.listdir(folder_name)}")
    if len(index_files) == 0:
        raise ValueError(f"No index file found for the folder {folder_name}")
    elif len(index_files) > 1:
        raise ValueError(f"Multiple index files found for the folder {folder_name}")
    else:
        index_file = os.path.join(folder_name, index_files[0])
        logging.info(f"Loading index file from: {index_file}")
        index = GPTSimpleVectorIndex.load_from_disk(index_file)
        response = index.query(input_text, response_mode="compact")
        return response.response

iface = gr.Interface(fn=lambda input_text: chatbot(folder_name, input_text), inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text", title="Gunbot Ultimate Language Model")
iface.launch(share=True)

if __name__ == "__main__":
    folder_name = 'gunthytrainingdata'
    index = construct_index(folder_name)
