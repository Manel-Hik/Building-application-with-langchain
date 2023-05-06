from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


load_dotenv()

# location of the pdf file/files. 
reader = PdfReader('./pdf_data/llama_paper.pdf')
#print(reader)

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

#print(raw_text[:100])

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
#print(len(texts))
#print(texts[0])

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

#take our text chunk and find its corresponding embedidng to be stored than in docsearch
docsearch = FAISS.from_texts(texts, embeddings)
#print(docsearch)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "who are the authors of the article?"
docs = docsearch.similarity_search(query)
#print(chain.run(input_documents=docs, question=query))


query = "What was the cost of training the llama model?"
docs = docsearch.similarity_search(query)
#print(chain.run(input_documents=docs, question=query))

query = "How was the model trained?"
docs = docsearch.similarity_search(query)
#print(chain.run(input_documents=docs, question=query))

query = "what was the size of the training dataset?"
docs = docsearch.similarity_search(query)
#print(chain.run(input_documents=docs, question=query))

query = "How is this different from other models?"
docs = docsearch.similarity_search(query)
#print(chain.run(input_documents=docs, question=query))


query = "whats the main difference between llama and chatgpt"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))