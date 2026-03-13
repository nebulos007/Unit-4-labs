import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import Language
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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

def load_with_fixed_size_chunking(vector_store, file_path):
    """
    Load a document by splitting it into fixed-size chunks using CharacterTextSplitter.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        file_path (str): Path to the file to load and chunk
    
    Returns:
        int: Total number of chunks successfully stored
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Create CharacterTextSplitter with fixed parameters
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separator=" "
        )
        
        # Split text into chunks (returns strings)
        text_chunks = splitter.split_text(content)
        
        # Create Document objects from chunks
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Load chunks into vector store
        file_name = os.path.basename(file_path)
        chunks_stored = load_document_with_chunks(vector_store, file_path, documents)
        
        # Print statistics
        if documents:
            avg_chunk_size = sum(len(chunk.page_content) for chunk in documents) / len(documents)
            print(f"📊 Chunking Statistics for '{file_name}':")
            print(f"   - Total chunks created: {len(documents)}")
            print(f"   - Average chunk size: {avg_chunk_size:.0f} characters\n")
        
        return chunks_stored
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        return 0
    
    except Exception as e:
        print(f"❌ Error chunking file '{file_path}': {str(e)}")
        return 0

def load_with_paragraph_chunking(vector_store, file_path):
    """
    Load a document by splitting it into chunks while preserving paragraphs.
    Uses RecursiveCharacterTextSplitter to intelligently split on paragraphs first,
    then newlines, then spaces.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        file_path (str): Path to the file to load and chunk
    
    Returns:
        int: Total number of chunks successfully stored
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Create RecursiveCharacterTextSplitter with paragraph-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text into chunks (returns strings)
        text_chunks = splitter.split_text(content)
        
        # Create Document objects from chunks
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Load chunks into vector store
        file_name = os.path.basename(file_path)
        chunks_stored = load_document_with_chunks(vector_store, file_path, documents)
        
        # Print comparison statistics
        if documents:
            chunk_sizes = [len(chunk.page_content) for chunk in documents]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            min_chunk_size = min(chunk_sizes)
            max_chunk_size = max(chunk_sizes)
            
            # Count chunks that start with newline (indicating paragraph preservation)
            paragraph_preserved = sum(1 for chunk in documents if chunk.page_content.startswith('\n'))
            
            print(f"📊 Paragraph-Aware Chunking Statistics for '{file_name}':")
            print(f"   - Total chunks created: {len(documents)}")
            print(f"   - Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   - Smallest chunk: {min_chunk_size} characters")
            print(f"   - Largest chunk: {max_chunk_size} characters")
            print(f"   - Chunks preserving paragraph boundaries: {paragraph_preserved}\n")
        
        return chunks_stored
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        return 0
    
    except Exception as e:
        print(f"❌ Error chunking file '{file_path}': {str(e)}")
        return 0

def load_with_markdown_structure_chunking(vector_store, file_path):
    """
    Load a markdown document by splitting on markdown headers while preserving structure.
    Uses MarkdownHeaderTextSplitter to split on headers, then RecursiveCharacterTextSplitter
    for final chunking with overlap.
    
    Args:
        vector_store: The InMemoryVectorStore instance
        file_path (str): Path to the markdown file to load and chunk
    
    Returns:
        int: Total number of chunks successfully stored
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Step 1: Split markdown by headers to preserve document structure
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        markdown_chunks = markdown_splitter.split_text(content)
        
        # Step 2: Apply RecursiveCharacterTextSplitter for final chunking with overlap
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200
        )
        
        # Split each markdown chunk further
        documents = []
        for chunk in markdown_chunks:
            split_docs = recursive_splitter.split_documents([chunk])
            documents.extend(split_docs)
        
        # Load chunks into vector store
        file_name = os.path.basename(file_path)
        chunks_stored = load_document_with_chunks(vector_store, file_path, documents)
        
        # Print statistics
        if documents:
            chunk_sizes = [len(doc.page_content) for doc in documents]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            
            print(f"📊 Markdown Structure-Aware Chunking Statistics for '{file_name}':")
            print(f"   - Total chunks created: {len(documents)}")
            print(f"   - Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   - Smallest chunk: {min(chunk_sizes)} characters")
            print(f"   - Largest chunk: {max(chunk_sizes)} characters")
            print(f"   - Chunk overlap: 200 characters for context preservation\n")
        
        return chunks_stored
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        return 0
    
    except Exception as e:
        print(f"❌ Error chunking file '{file_path}': {str(e)}")
        return 0

def create_search_tool(vector_store):
    """
    Create a LangChain Tool for searching the company document repository.
    
    Args:
        vector_store: The InMemoryVectorStore instance containing indexed documents
    
    Returns:
        A LangChain Tool that agents can use for semantic search
    """
    
    @tool
    def search_documents(query: str) -> str:
        """
        Searches the company document repository for relevant information based on the given query.
        Use this to find information about company policies, benefits, and procedures.
        
        Args:
            query: The search query string
        
        Returns:
            Formatted search results with similarity scores
        """
        # Perform similarity search
        results = vector_store.similarity_search_with_score(query, k=3)
        
        # Format results
        if not results:
            return "No relevant documents found."
        
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            formatted_results.append(
                f"Result {i} (Score: {score:.4f}): {doc.page_content}"
            )
        
        return "\n\n".join(formatted_results)
    
    return search_documents

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

    # Create ChatOpenAI instance with GitHub Models API configuration
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN")
    )

    # Display header for document loading
    print("=== Loading Documents into Vector Database ===\n")

    # Load the HealthInsuranceBrochure document
    document_file = "HealthInsuranceBrochure.md"
    doc_id = load_document(vector_store, document_file)
    
    if doc_id:
        print(f"✅ Document '{document_file}' successfully indexed in vector store\n")
    else:
        print(f"⚠️ Failed to load document. Continuing...\n")

    # Load the EmployeeHandbook document with markdown structure-aware chunking
    document_file = "EmployeeHandbook.md"
    chunks_stored = load_with_markdown_structure_chunking(vector_store, document_file)
    
    if chunks_stored > 0:
        print(f"✅ Document '{document_file}' successfully chunked and indexed in vector store\n")
    else:
        print(f"⚠️ Failed to load document. Continuing...\n")

   # =================================================================
    # ReAct Pattern Agent Setup
    # =================================================================
    print("=== Initializing ReAct Agent ===\n")
    
    # Create the search tool from the vector store
    search_tool = create_search_tool(vector_store)
    search_tools = [search_tool]
    
    # Create the prompt template with ReAct pattern
    # System message provides the agent's role and guidelines
    # MessagesPlaceholder for chat_history maintains conversation context
    # user input placeholder for the question
    # {agent_scratchpad} for tracking the agent's reasoning process (as a string, not a message list)
    # Note: {tools} and {tool_names} are required placeholders for create_react_agent
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions about company policies, benefits, and procedures. "
            "You have access to the following tools:\n\n{tools}\n\n"
            "Use the following format:\n"
            "Thought: Do I need to use a tool? Yes\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Always cite which document chunks you used in your answer."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "{agent_scratchpad}"),
    ])
    
    # Create the ReAct agent using the LLM, tools, and prompt template
    # ReAct pattern: Reason and Act iteratively
    # The agent thinks about what to do, then executes an action
    agent = create_react_agent(
        llm=chat_model,
        tools=search_tools,
        prompt=prompt
    )
    
    # Create the AgentExecutor to run the agent
    # AgentExecutor orchestrates the agent's reasoning and tool execution
    # verbose=False to reduce unnecessary output of search results
    # We'll show only the final answer, not intermediate tool calls
    agent_executor = AgentExecutor(
        agent=agent,
        tools=search_tools,
        verbose=False,  # Set to False to hide intermediate tool outputs
        handle_parsing_errors=True
    )
    
    print("✅ ReAct Agent initialized successfully\n")
    print(f"🔧 Available tools: {[tool.name for tool in search_tools]}\n")
    
    # =================================================================
    # Interactive Chat Interface
    # =================================================================
    print("=== Conversational Agent Chat Interface ===")
    print("Welcome to the Company Information Assistant!")
    print("Ask questions about company policies, benefits, and procedures.")
    print("The agent will search the company documents to find relevant information.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Initialize chat history to maintain conversation context
    chat_history = []
    
    # Maximum number of message pairs to keep in history (to manage token limits)
    # Reduced aggressively to prevent token overflow with gpt-4o's 8000 token limit
    # 4 messages = 2 message pairs (last question + answer pair)
    # This ensures we stay well under the token limit even with large search results
    MAX_HISTORY_LENGTH = 4
    
    # Interactive chat loop
    while True:
        # Prompt user for input
        user_input = input("You: ").strip()
        
        # Exit conditions
        if user_input.lower() in ["quit", "exit"]:
            print("Agent: Thank you for using the Company Information Assistant. Goodbye!")
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        try:
            # Invoke the agent with user input and chat history
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Extract the agent's response
            agent_response = response.get("output", "No response generated.")
            
            # Print the agent's response
            print(f"\nAgent: {agent_response}\n")
            
            # Add the exchange to chat history for context in future turns
            # Using HumanMessage and AIMessage from langchain_core.messages
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))
            
            # Implement a sliding window to manage token limits
            # Keep only the most recent messages to prevent exceeding token limits
            if len(chat_history) > MAX_HISTORY_LENGTH:
                chat_history = chat_history[-MAX_HISTORY_LENGTH:]
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            print("Please try again with a different question.\n")

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
