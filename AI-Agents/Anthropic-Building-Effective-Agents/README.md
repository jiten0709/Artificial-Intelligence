# Building Effective AI Agents: Practical Patterns

This repository provides Python implementations of the powerful agentic design patterns discussed in Anthropic's "Building effective agents" engineering blog post. Each script is a self-contained, practical example demonstrating a specific pattern for creating more capable and reliable AI systems.

**Reference:** [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

## 🚀 Agentic Patterns Implemented

This project explores the following key patterns for building sophisticated AI workflows:

1.  **Prompt Chaining (`1-prompt-chaining.py`)**: A fundamental pattern where the output of one prompt is used as the input for the next, creating a simple, sequential workflow.

2.  **Routing (`2-routing.py`)**: Demonstrates how to use an AI model to act as a "router," intelligently deciding which tool, function, or next step to take based on the input.

3.  **Parallelization (`3-parallelization.py`)**: Shows how to execute multiple independent AI calls concurrently, significantly speeding up workflows that involve several non-dependent tasks.

4.  **Orchestrator-Worker (`4-orchestrator-workers.py`)**: A powerful hierarchical pattern where a central "orchestrator" agent breaks down a complex task and delegates sub-tasks to specialized "worker" agents. This example builds a complete blog post by coordinating research, writing, and editing agents.

5.  **Evaluator-Optimizer (`5-evaluator-optimizer.py`)**: Implements a self-correcting loop where one agent ("evaluator") critiques the work of another, and an "optimizer" agent refines the output based on the feedback.

## 🛠️ Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- An OpenAI API Key

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/jiten0709/Artificial-Intelligence.git
    cd Artificial-Intelligence
    ```

2.  **Create a virtual environment (recommended):**

    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a file named `.env` and refer `.env.example`.
2.  Add your OpenAI API key to the `.env` file:

## ▶️ How to Run

Each Python script is a standalone example of a pattern. You can run any of them directly from your terminal.

For example, to run the Orchestrator-Worker pattern:

```sh
python 4-orchestrator-workers.py
```

To run the Evaluator-Optimizer pattern:

```sh
python 5-evaluator-optimizer.py
```

### Mock Mode

To test the logic of the scripts without consuming API credits, you can set `USE_MOCK_MODE="true"` in your `.env` file. The scripts will output mock data instead of calling the OpenAI API.

## ✨ Example In-Depth: Orchestrator-Worker Pattern

The `4-orchestrator-workers.py` script provides a clear example of building a complex system.

- **Goal**: To write a comprehensive blog post on a given topic.
- **Roles**:
  - **Orchestrator**: Breaks the blog post into sections (e.g., introduction, main points, conclusion).
  - **Worker**: Writes the content for each individual section, based on the orchestrator's plan.
  - **Reviewer**: Reads the complete draft, scores it for cohesion, and provides a final polished version.
- **Output**: The final, polished blog post is saved to `final_blog_post.txt`.

---

This repository serves as a practical toolkit for developers looking to move beyond simple prompts and build more robust, modular, and effective AI-powered applications.

## ~ By Jiten 🥰
