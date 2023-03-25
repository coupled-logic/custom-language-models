import json
import logging
from gpt_index import GPTSimpleVectorIndex, PromptHelper
from gpt_index.llm_predictor import LLMPredictor
from langchain import OpenAI

from gpt_index.indices.vector_store.base import BaseDocument
import os
print(os.getcwd())

YOUR_API_KEY = "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"

class SingleFileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        documents = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    documents.append(CustomDocument(**data))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            documents.append(CustomDocument(**item))
            except ValueError as e:
                logging.error(f"Error loading {self.file_path}: {str(e)}")
        return documents


def construct_index(json_file_path):

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 100
    chunk_size_limit = 300

    # define prompt
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            openai_api_key='YOUR_API_KEY',
            temperature="0.0",
            model_name="text-davinci-003",
            max_tokens=num_outputs
        ))
    logging.info(f"Constructing index for JSON file {json_file_path}...")
    # load data — it will take a single JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                documents = [CustomDocument(**data)]
            elif isinstance(data, list):
                documents = []
                for item in data:
                    if isinstance(item, dict):
                        documents.append(CustomDocument(**item))
        except ValueError as e:
            logging.error(f"Error loading {json_file_path}: {str(e)}")
            return

    # create vector index
    vector_index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper, document_class=CustomDocument
    )
    # vector_index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    vector_index.save_to_disk(f"/vector_index_{timestamp}.json")

    # print document lengths
    total_length = 0
    for doc in base_documents:
        doc_length = len(json.dumps(doc))
        total_length += doc_length
        print(f"Document length: {doc_length}")

    print(f"Total length of documents: {total_length}")

    file_path = 'docs/midjourney/merged_data.json'
    reader = SingleFileReader(file_path)
    documents = reader.load_data()

