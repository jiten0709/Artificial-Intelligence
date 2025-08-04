# LangChain Library Examples

This directory contains a collection of Jupyter notebooks and Python scripts that demonstrate various features, patterns, and capabilities of the LangChain library. Each sub-directory focuses on a specific aspect of LangChain.

## 📂 Directory Structure

- **`/Chains`**: Explores the LangChain Expression Language (LCEL) to build both simple and complex processing pipelines.
- **`/RAG`**: Provides examples of Retrieval-Augmented Generation (RAG) systems, from basic to production-ready.
- **`/Tool`**: Demonstrates how to create and use tools within LangChain agents to interact with external systems.
- **`/Save-Chat-History`**: Shows how to persist conversation history with external databases.

---

## ✨ Key Examples

### ⛓️ Chains

The notebooks in the [`Chains`](Tool-Framework-Library/Langchain/Chains) directory showcase different ways to construct data processing flows:

- **[`helloworld.ipynb`](Tool-Framework-Library/Langchain/Chains/helloworld.ipynb)**: An introduction to creating a basic chain with `RunnableSequence`.
- **[`sequential_chaining.ipynb`](Tool-Framework-Library/Langchain/Chains/sequential_chaining.ipynb)**: Demonstrates how to pipe the output of one model into the input of another to create a multi-step sequence.
- **[`parallel_chaining.ipynb`](Tool-Framework-Library/Langchain/Chains/parallel_chaining.ipynb)**: Shows how to use `RunnableParallel` to execute multiple chains concurrently and combine their results.
- **[`conditional_chaining.ipynb`](Tool-Framework-Library/Langchain/Chains/conditional_chaining.ipynb)**: Implements conditional logic to dynamically route the flow of a chain based on the input.

### 📚 Retrieval-Augmented Generation (RAG)

The [`RAG`](Tool-Framework-Library/Langchain/RAG) directory contains examples of building Q&A systems over custom documents:

- **[`helloworld.ipynb`](Tool-Framework-Library/Langchain/RAG/helloworld.ipynb)**: A fundamental RAG implementation that processes a single text file, stores embeddings in a Chroma vector database, and answers questions based on the retrieved context.
- **[`helloworld_with_metadata.ipynb`](Tool-Framework-Library/Langchain/RAG/helloworld_with_metadata.ipynb)**: A more advanced, production-ready RAG system that handles multiple documents, enriches data with metadata, and provides a robust interactive Q&A session with detailed logging and source tracking.

### 🛠️ Tools

The [`Tool`](Tool-Framework-Library/Langchain/Tool) directory demonstrates how to empower agents with custom tools:

- **[`helloworld.ipynb`](Tool-Framework-Library/Langchain/Tool/helloworld.ipynb)**: Creates a ReAct agent that can use a custom tool to fetch the current time for different timezones, showcasing how agents can leverage external functions to answer questions they cannot answer on their own.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A Google API Key for Gemini models.

### Configuration

1.  Create a `.env` file in the [`Tool-Framework-Library/Langchain`](Tool-Framework-Library/Langchain) directory.
2.  Add your Google API key to the `.env` file, as shown in the example below.

    ```env
    # .env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

### Running the Examples

Navigate to the desired subdirectory and open the Jupyter notebooks (`.ipynb` files). You can run the cells sequentially to see the code in action. Most notebooks include package installation commands for any required dependencies.
