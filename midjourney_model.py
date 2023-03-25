import os
import json
import logging
from gpt_index import GPTSimpleVectorIndex, PromptHelper
from gpt_index.llm_predictor import LLMPredictor
from langchain import OpenAI

from gpt_index.indices.vector_store.base import BaseDocument

folder_path = str('/docs/midjourney')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


class SimpleDirectoryReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_data(self):
        documents = []
        for file in os.listdir(self.folder_path):
            print(os.listdir(self.folder_path))
            if file.endswith(".json"):
                with open(os.path.join(self.folder_path, file), "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            documents.append(CustomDocument(**data))
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    documents.append(CustomDocument(**item))
                    except ValueError as e:
                        logging.error(f"Error loading {file}: {str(e)}")
        return documents



class CustomDocument(BaseDocument):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def text(self):
        return self.__dict__.get('text', '')

    @property
    def attributes(self):
        return self.__dict__

    def get_type(self):
        return "custom"



def construct_index(folder_path):

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
    logging.info(f"Constructing index for folder {folder_path}...")
    # load data — it will take all the .txt/json files, if there are more than 1
    reader = SimpleDirectoryReader(folder_path)
    print(reader)

    documents = reader.load_data()
    base_documents = []

    for document in documents:
        if isinstance(document, list):
            base_documents.extend(document)
        elif isinstance(document, dict):
            base_documents.append(document)

    # create vector index
    vector_index = GPTSimpleVectorIndex(
        base_documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper, document_class=CustomDocument
    )
    # vector_index = GPTSimpleVectorIndex(base_documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    vector_index.save_to_disk(f"/vector_index_{timestamp}.json")

    # print document lengths
    total_length = 0
    for doc in base_documents:
        doc_length = len(json.dumps(doc))
        total_length += doc_length
        print(f"Document length: {doc_length}")

    print(f"Total length of documents: {total_length}")


if __name__ == '__main__':
    OPEN_AI_API_KEY = "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"
    folder_path = 'docs/midjourney'
    construct_index(folder_path)











#
# import os
# import json
# import logging
# from gpt_index import GPTSimpleVectorIndex, PromptHelper
# from gpt_index.llm_predictor import LLMPredictor
# from langchain import OpenAI
#
#
# class SimpleDirectoryReader:
#     def __init__(self, folder_path):
#         self.folder_path = folder_path
#
#     def load_documents(self):
#         documents = []
#         for file in os.listdir(self.folder_path):
#             if file.endswith(".json"):
#                 with open(os.path.join(self.folder_path, file), "r", encoding="utf-8") as f:
#                     try:
#                         document = f.read()
#                         if document:
#                             documents.append((file, len(json.loads(document.strip()))))
#                     except ValueError as e:
#                         logging.error(f"Error loading {file}: {str(e)}")
#         return documents
#
#
# def construct_index(folder_path):
#     OPEN_AI_API_KEY = "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"
#     max_input_size = 4096
#     num_outputs = 512
#     max_chunk_overlap = 100
#     chunk_size_limit = 300
#
#     try:
#         # define prompt
#         prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#         # define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
#         llm_predictor = LLMPredictor(
#             llm=OpenAI(
#                 api_key=OPEN_AI_API_KEY,
#                 temperature="0.0",
#                 model_name="text-davinci-003",
#                 max_tokens=num_outputs
#             ))
#
#         # load data — it will take all the .json files
#         reader = SimpleDirectoryReader(folder_path)
#         base_documents = [json.loads(text) for file, text in reader.load_documents()]
#
#         # create vector index
#         vector_index = GPTSimpleVectorIndex(base_documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#         vector_index.save_to_disk()
#     except Exception as e:
#         logging.error(f"Error during index construction: {str(e)}")
#
#
# folder_path = 'docs/midjourney'
# logging.info(f"Constructing index for folder {folder_path}...")
# construct_index(folder_path)
#
# # load data — it will take all the .txt/json files, if there are more than 1
#
#




