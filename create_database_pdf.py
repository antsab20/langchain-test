from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH_PDF = "chroma_pdf"

loader = PyPDFLoader("pdfs/vger.pdf")
pages = loader.load_and_split()
db = Chroma.from_documents(pages,
        OpenAIEmbeddings(), persist_directory=CHROMA_PATH_PDF)
db.persist()