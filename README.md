# AI Video Assistant with RAG

An AI-powered video assistant built using Python, Streamlit, LangChain, Groq AI, Whisper, and ChromaDB. The application transcribes video content, generates summaries, extracts key insights, and answers user questions using Retrieval-Augmented Generation (RAG).

## Problem Statement

Understanding lengthy videos, lectures, and tutorials can be time-consuming. Finding specific information often requires watching the video again or manually searching through its content.

The AI Video Assistant with RAG addresses this problem by converting video content into text, generating concise summaries, and allowing users to ask questions about the content using AI-powered retrieval.

## Features

- Video Audio Extraction
- Speech-to-Text Transcription
- AI-Powered Video Summarization
- Interactive Question Answering
- Retrieval-Augmented Generation (RAG)
- Transcript Chunking and Embeddings
- Semantic Search Using ChromaDB
- Key Insights Extraction
- Context-Aware AI Responses
- Interactive Streamlit Interface

## Architecture

Video Input ↓ Audio Extraction ↓ Speech-to-Text Transcription ↓ Transcript Processing & Chunking ↓ Embedding Generation ↓ ChromaDB Vector Storage ↓ User Question ↓ Relevant Context Retrieval ↓ LLM Response Generation

## Tech Stack

### Backend & Application

- Python
- Streamlit

### AI & LLM Integration

- Groq API
- LangChain
- Retrieval-Augmented Generation (RAG)

### Transcription & Video Processing

- OpenAI Whisper
- yt-dlp
- FFmpeg

### Vector Database & Retrieval

- ChromaDB
- Sentence Transformers
- Text Embeddings

## Workflow

1. User provides supported video content.
2. The application extracts audio and transcribes the speech into text.
3. The transcript is processed and divided into smaller chunks.
4. Text embeddings are generated and stored in ChromaDB.
5. The user asks a question about the video.
6. The system retrieves relevant transcript chunks using semantic search.
7. The LLM generates a response using the retrieved context.

## Benefits & Use Cases

- **Time Saving:** Understand lengthy videos without watching them completely.
- **Educational Support:** Summarize lectures and revise important concepts.
- **Quick Information Retrieval:** Find answers from video content through natural-language questions.
- **Improved Productivity:** Extract useful information from tutorials and training videos.
- **Context-Aware Responses:** Generate answers grounded in relevant transcript content.

## Output

- Video Transcript
- AI-Generated Summary
- Key Insights
- Context-Aware Question Answers
- Relevant Information Retrieved from the Transcript

## Future Improvements

- Timestamp-Based Answers
- Exportable Summaries and Study Notes
- Enhanced Multilingual Support
- Support for Additional Video Sources
- Improved Long-Video Processing

## Author

**Mohd. Faizan Khan**