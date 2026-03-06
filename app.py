import os
import math
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

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

    # Print lab header
    print("=== Embedding Inspector Lab ===")
    print("Generating embeddings for three sentences...\n")

    # Test sentences for embedding demonstration
    test_sentences = [
        "The canine barked loudly.",
        "The dog made a noise.",
        "The electron spins rapidly."
    ]

    # Create metadata for each sentence
    metadatas = [
        {
            "created_at": datetime.now().isoformat(),
            "index": i
        }
        for i in range(len(test_sentences))
    ]

    # Add sentences to vector store with metadata
    vector_store.add_texts(test_sentences, metadatas=metadatas)

    # Print confirmation and display stored sentences
    print(f"✅ Successfully stored {len(test_sentences)} sentences in vector store\n")
    for i, sentence in enumerate(test_sentences):
        print(f"Sentence {i+1}: {sentence}")

    # Generate embeddings for cosine similarity analysis
    embedding_vectors = [embeddings.embed_query(sentence) for sentence in test_sentences]

    # Calculate and display cosine similarity between sentence pairs
    print("\n=== Cosine Similarity Analysis ===\n")

    # Similarity between Sentence 1 and Sentence 2
    similarity_1_2 = cosine_similarity(embedding_vectors[0], embedding_vectors[1])
    print(f"Sentence 1 vs Sentence 2: {similarity_1_2:.4f}")

    # Similarity between Sentence 2 and Sentence 3
    similarity_2_3 = cosine_similarity(embedding_vectors[1], embedding_vectors[2])
    print(f"Sentence 2 vs Sentence 3: {similarity_2_3:.4f}")

    # Similarity between Sentence 3 and Sentence 1
    similarity_3_1 = cosine_similarity(embedding_vectors[2], embedding_vectors[0])
    print(f"Sentence 3 vs Sentence 1: {similarity_3_1:.4f}")

    # Interactive semantic search loop
    print("\n=== Semantic Search ===")
    
    while True:
        query = input("\nEnter a search query (or 'quit' to exit): ").strip()
        
        # Check for exit commands
        if query.lower() in ["quit", "exit"]:
            break
        
        # Skip empty queries
        if not query:
            continue
        
        # Perform semantic search
        search_sentences(vector_store, query)
    
    print("\n👋 Goodbye! Thanks for using the Embedding Inspector Lab.\n")


if __name__ == "__main__":
    main()
