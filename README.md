# Rag-Based-AI-Teaching-Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)

An interactive, voice-first AI Teaching Assistant built with an end-to-end Retrieval-Augmented Generation (RAG) pipeline. This application allows users to dynamically ingest YouTube lecture videos, builds a searchable vector knowledge base, and provides highly accurate, context-aware answers using local LLMs. 

## ✨ Key Features

* **Dynamic Knowledge Ingestion:** Paste any YouTube lecture link to automatically extract the audio track (via `yt-dlp`), transcribe it using **OpenAI Whisper**, and index the semantic chunks into the vector database.
* **Voice-First Interaction:** Integrated browser microphone support. Ask questions out loud; the system normalizes the audio via `FFmpeg`, transcribes it locally, and feeds it to the RAG pipeline.
* **Text-to-Speech (TTS) Autoplay:** The AI assistant speaks its answers back to you in real-time using `gTTS` and dynamic HTML audio injection.
* **Intelligent Deep-Linking:** AI responses include clickable UI buttons that deep-link directly to the exact timestamp in the source YouTube video where the topic was discussed.
* **100% Local Privacy & Zero Cost:** All embeddings and LLM inferences are run locally via **Ollama**, ensuring complete data privacy and zero API costs.

## 🧠 System Architecture

1. **Extraction:** `yt-dlp` extracts a lightweight audio stream from a YouTube URL.
2. **Transcription & Chunking:** `FFmpeg` normalizes the audio (16kHz mono WAV), and `Whisper (small)` transcribes the text with precise timestamps.
3. **Embedding:** `BGE-M3` (via Ollama) converts text chunks into semantic vector embeddings.
4. **Vector Storage:** Data is stored in a structured Pandas DataFrame and serialized via `joblib` for persistent session memory.
5. **Retrieval:** Cosine similarity (`scikit-learn`) matches the user's query against the vector matrix to find the top-K relevant video chunks.
6. **Generation:** `Llama 3.2` processes the retrieved context and generates a synthesized, teacher-like response.

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Audio-Recorder-Streamlit
* **AI Models:** Llama 3.2 (Text Gen), BGE-M3 (Embeddings), Whisper (Speech-to-Text)
* **Audio Processing:** yt-dlp, FFmpeg, gTTS
* **Data Processing:** Pandas, NumPy, Scikit-learn, Joblib

## 🚀 Installation & Setup

### Prerequisites
* Python 3.9+
* [FFmpeg](https://ffmpeg.org/download.html) installed and added to system PATH.
* [Ollama](https://ollama.com/) installed and running locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/ai-teaching-assistant.git](https://github.com/yourusername/ai-teaching-assistant.git)
cd ai-teaching-assistant