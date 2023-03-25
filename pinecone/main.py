import pinecone
import openai

pinecone.init(api_key="f4d7776c-7ebc-4364-9d1a-3acb890de2d4", environment="us-west4-gcp")
index = pinecone.Index("gunthy-wiki-web")
)

# check if 'openai' index already exists (only create index if not)
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
# connect to index
# check if 'openai' index already exists (only create index if not)
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
# connect to index
index = pinecone.Index('openai')
from datasets import load_dataset

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')

from tqdm.auto import tqdm  # this is our progress bar

batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))

openai.api_key = OPENAI_API_KEY
response = openai.Embedding.create(
    input="What caused the 1929 Great Depression?",
    model="text-similarity-babbage-001"
)

res = index.query(
    vector=response['data'][0]['embedding'], top_k=5, include_values=True
)
for match in res['results'][0]['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")