import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma_pdf"
PROMPT_TEMPLATE = """
{history}

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 3

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Initialize conversation history
conversation_history = ""

def search_database(query, db):
    try:
        results = db.similarity_search_with_relevance_scores(query, k=TOP_K_RESULTS)
        return results
    except Exception as e:
        print(f"Error during database search: {e}")
        return []

def generate_response(query, results, history):
    if len(results) == 0 or results[0][1] < SIMILARITY_THRESHOLD:
        return "Unable to find matching results."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(history=history, context=context_text, question=query)

    model = ChatOpenAI()
    response = model.invoke(prompt)

    response_text = response.content if hasattr(response, 'content') else "No response generated."
    return response_text

if __name__ == "__main__":
    print("LangChain Chat Interface. Type 'exit' to leave the chat.")
    while True:
        query_text = input("You: ")
        if query_text.lower() == 'exit':
            break

        results = search_database(query_text, db)
        response = generate_response(query_text, results, conversation_history)
        print("Bot:", response)

        conversation_history += f"You: {query_text}\nBot: {response}\n"
