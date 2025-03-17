from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Load data
loader = TextLoader(r"C:\Users\asush\Downloads\plants_diseases.txt")
documents = loader.load()

# Embedding
embeddings = OpenAIEmbeddings()

# LLM config - you might still need OpenAI LLM, 
# but will rely on retrieval outputs
llm = OpenAI(
    temperature=0.0,  # Lower temperature to minimize generation randomness
)

# Create a RetrievalQA chain strictly using the vectorstore
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # Use map_reduce to focus on document aggregation
    retriever=vectorstore.as_retriever(),
)

# Function to get plant disease information based on a query
def get_disease_info(query):
    """
    Gets information about a plant disease based on a user's query.

    Args:
        query (str): The user's query about the disease.

    Returns:
        str: A string containing information in response to the query.
    """
    result = qa_chain.invoke({"query": query})
    return result['result']

# Interactive loop for user input
if __name__ == "__main__":
    initial_disease_name = input("Enter the name of the plant disease: ")
    initial_query = f"What are the symptoms, causes, prevention, and treatment for {initial_disease_name}?"
    disease_info = get_disease_info(initial_query)
    print("\nDisease Information:")
    print(disease_info)
    
    while True:
        follow_up_question = input("\nDo you have any follow-up questions about this disease? (just press enter to exit): ").strip()
        if not follow_up_question:
            break
        follow_up_info = get_disease_info(follow_up_question)
        print("\nAdditional Information:")
        print(follow_up_info)