from langchain_community.document_loaders import TextLoader


loader = TextLoader('my.txt')

docs= loader.load()

print(len(docs))

print(docs[0])