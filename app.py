import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import Language

# Load environment variables
load_dotenv()

def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    Cosine similarity = (A · B) / (||A|| * ||B||)
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    
    return dot_product / (norm_a * norm_b)

def search_sentences(vector_store, query, k=3):
    """
    Search for similar sentences in the vector store.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        query (str): The search query string
        k (int): Number of results to return (default: 3)
    
    Returns:
        list: List of tuples containing (document, similarity_score)
    """
    # Perform similarity search with scores
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # Print search results with formatting
    print(f"\n🔍 Search Results for: \"{query}\"\n")
    
    for rank, (document, score) in enumerate(results, 1):
        print(f"{rank}. Similarity: {score:.4f} | {document.page_content}")
    
    return results

def load_document(vector_store, file_path):
    """
    Load a document from a file and add it to the vector store.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        file_path (str): Path to the file to load
    
    Returns:
        str: The document ID or message indicating success
    
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For other file reading errors
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract file name from path
        file_name = os.path.basename(file_path)
        
        # Create a Document object with metadata
        document = Document(
            page_content=content,
            metadata={
                "fileName": file_name,
                "createdAt": datetime.now().isoformat()
            }
        )
        
        # Add document to vector store
        doc_ids = vector_store.add_documents([document])
        
        # Print success message
        print(f"✅ Loaded '{file_name}' ({len(content)} characters)")
        
        # Return the document ID
        return doc_ids[0] if doc_ids else "Document added"
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        return None
    
    except Exception as e:
        error_message = str(e).lower()
        
        # Check if error is related to document size/token limits
        if "maximum context length" in error_message or "token" in error_message:
            print(f"❌ Error loading '{os.path.basename(file_path)}':")
            print("⚠️ This document is too large to embed as a single chunk.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: The document needs to be split into smaller chunks.")
        else:
            print(f"❌ Error loading file '{file_path}': {str(e)}")
        return None

def load_document_with_chunks(vector_store, file_path, chunks):
    """
    Load document chunks into the vector store with updated metadata.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        file_path (str): Path to the original document file
        chunks (list): List of LangChain Document objects (chunks)
    
    Returns:
        int: Total number of chunks successfully stored
    
    Raises:
        Exception: For document storage errors
    """
    try:
        file_name = os.path.basename(file_path)
        total_chunks = len(chunks)
        chunks_stored = 0
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            # Update metadata for each chunk
            chunk.metadata.update({
                "fileName": f"{file_name} (Chunk {i}/{total_chunks})",
                "createdAt": datetime.now().isoformat(),
                "chunkIndex": i
            })
            
            # Add chunk to vector store
            doc_ids = vector_store.add_documents([chunk])
            
            if doc_ids:
                chunks_stored += 1
                print(f"✅ Loaded chunk {i}/{total_chunks} from '{file_name}' ({len(chunk.page_content)} characters)")
            else:
                print(f"⚠️ Failed to store chunk {i}/{total_chunks}")
        
        print(f"\n✅ Successfully stored {chunks_stored}/{total_chunks} chunks\n")
        return chunks_stored
    
    except Exception as e:
        error_message = str(e).lower()
        
        # Check if error is related to document size/token limits
        if "maximum context length" in error_message or "token" in error_message:
            print(f"❌ Error loading chunks from '{os.path.basename(file_path)}':")
            print("⚠️ One or more chunks exceeded the token limit.")
            print("Token limit exceeded. The embedding model can only process up to 8,191 tokens at once.")
            print("Solution: Reduce the chunk_size parameter for smaller chunks.")
        else:
            print(f"❌ Error loading chunks from '{file_path}': {str(e)}")
        
        return 0

def main():
    print("🤖 Python LangChain Agent Starting...\n")

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("Please create a .env file with your GitHub token:")
        print("GITHUB_TOKEN=your-github-token-here")
        print("\nGet your token from: https://github.com/settings/tokens")
        print("Or use GitHub Models: https://github.com/marketplace/models")
        return

    # Create OpenAIEmbeddings instance with GitHub Models API configuration
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        check_embedding_ctx_length=False
    )

    # Create InMemoryVectorStore instance
    vector_store = InMemoryVectorStore(embeddings)

    # Display header for document loading
    print("=== Loading Documents into Vector Database ===\n")

    # Load the HealthInsuranceBrochure document
    document_file = "HealthInsuranceBrochure.md"
    doc_id = load_document(vector_store, document_file)
    
    if doc_id:
        print(f"✅ Document '{document_file}' successfully indexed in vector store\n")
    else:
        print(f"⚠️ Failed to load document. Continuing...\n")

    # Load the EmployeeHandbook document
    document_file = "EmployeeHandbook.md"
    doc_id = load_document(vector_store, document_file)
    
    if doc_id:
        print(f"✅ Document '{document_file}' successfully indexed in vector store\n")
    else:
        print(f"⚠️ Failed to load document. Continuing...\n")

    # Print lab header
    print("=== Embedding Inspector Lab ===")
    print("Loading documents from file system...\n")

    # TODO: Replace with document loading logic
    # # Test sentences for embedding demonstration - diverse topics
    # test_sentences = [
    #     # Animals and pets
    #     "The canine barked loudly.",
    #     "The dog made a noise.",
    #     "My cat loves to sleep on the windowsill.",
    #     "The bird sang a beautiful morning song.",
    #     
    #     # Science and physics
    #     "The electron spins rapidly.",
    #     "Quantum mechanics studies subatomic particles.",
    #     "Gravity pulls objects toward the Earth.",
    #     
    #     # Food and cooking
    #     "I made a delicious pasta for dinner.",
    #     "The chef prepared a gourmet meal with fresh ingredients.",
    #     "Chocolate cake is a classic dessert.",
    #     
    #     # Sports and activities
    #     "The basketball player scored a three-pointer.",
    #     "Running marathons requires endurance and training.",
    #     
    #     # Weather and nature
    #     "The storm brought heavy rain and thunder.",
    #     "Sunsets paint the sky with vibrant orange colors.",
    #     
    #     # Technology and programming
    #     "Python is a popular programming language.",
    #     "Artificial intelligence is transforming technology."
    # ]

    # TODO: Replace with document loading and storage logic
    # # Create metadata for each sentence
    # metadatas = [
    #     {
    #         "created_at": datetime.now().isoformat(),
    #         "index": i
    #     }
    #     for i in range(len(test_sentences))
    # ]
    #
    # # Add sentences to vector store with metadata
    # vector_store.add_texts(test_sentences, metadatas=metadatas)
    #
    # # Print confirmation and display stored sentences
    # print(f"✅ Successfully stored {len(test_sentences)} sentences in vector store\n")
    # for i, sentence in enumerate(test_sentences):
    #     print(f"Sentence {i+1}: {sentence}")
    #
    # # Generate embeddings for cosine similarity analysis
    # embedding_vectors = [embeddings.embed_query(sentence) for sentence in test_sentences]
    #
    # # Calculate and display cosine similarity between sentence pairs
    # print("\n=== Cosine Similarity Analysis ===\n")
    #
    # # Similarity between Sentence 1 and Sentence 2
    # similarity_1_2 = cosine_similarity(embedding_vectors[0], embedding_vectors[1])
    # print(f"Sentence 1 vs Sentence 2: {similarity_1_2:.4f}")
    #
    # # Similarity between Sentence 2 and Sentence 3
    # similarity_2_3 = cosine_similarity(embedding_vectors[1], embedding_vectors[2])
    # print(f"Sentence 2 vs Sentence 3: {similarity_2_3:.4f}")
    #
    # # Similarity between Sentence 3 and Sentence 1
    # similarity_3_1 = cosine_similarity(embedding_vectors[2], embedding_vectors[0])
    # print(f"Sentence 3 vs Sentence 1: {similarity_3_1:.4f}")


if __name__ == "__main__":
    main()
