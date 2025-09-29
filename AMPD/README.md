# BESS Support Chatbot - README

This project provides an AI-powered chat experience to help users with their Battery Energy Storage Systems (BESS) by answering questions based on the "AmpD Enertainer User Manual".

It uses a **Multi-Modal RAG** architecture to intelligently understand both text and images from the manual.

## Features

- **Multi-Modal Understanding**: Extracts and analyzes both text and images for comprehensive context.
- **Accurate & Grounded**: Answers are generated *only* from the provided manual to prevent incorrect information.
- **Efficient**: Pre-processes the document and saves embeddings to a local vector store for fast, subsequent queries.
- **Modular Code**: The code is well-structured, commented, and easy to understand.

## Technology Stack

- **Language**: Python 3.11+
- **AI Framework**: LangChain
- **LLM & Embeddings**: OpenAI API (`gpt-4o`, `text-embedding-3-small`)
- **PDF Processing**: `PyMuPDF`
- **Vector DB**: `ChromaDB`

## Setup Instructions

**1. Prerequisites:**
   - Python 3.11 or newer.
   - The user manual PDF file `AmpD Enertainer User Manual (NCM) - Rev 2.3.pdf` must be in the same directory as the script.

**2. Clone or Download the Project:**
   - Download `bess_chatbot.py`, `requirements.txt`, and this `README.md` file.
   - Place the user manual PDF in the same folder.

**3. Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`