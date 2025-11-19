from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('Interim.pdf')

docs= loader.load()

print(docs[5].metadata)

