import os
import json
import requests
import subprocess
import numpy as np
import pandas as pd
import joblib
import whisper
import yt_dlp
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import torch
import base64
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

# --- PAGE CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="Sigma AI Teaching Assistant", page_icon="🎓", layout="wide")
st.title("🎓 RAG-Based AI Teaching Assistant")
st.caption("Enhance your knowledge base dynamically and ask context-aware questions.")

EMBEDDINGS_FILE = 'embeddings.joblib'

# Helper function to load or create the core vector store tracking matrix
def load_vector_store():
    if os.path.exists(EMBEDDINGS_FILE):
        return joblib.load(EMBEDDINGS_FILE)
    else:
        # Returns an empty dataframe matching your exact record keys if file doesn't exist yet
        return pd.DataFrame(columns=["number", "title", "start", "end", "text", "chunk_id", "embedding"])

# Load data into memory once and preserve it across interactions via Streamlit session state
if "df" not in st.session_state:
    st.session_state.df = load_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CORE PROCESSING UTILITIES (YOUR EXACT PIPELINE) ---

def create_embedding(text_list):
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list}
        )
        data = r.json()
        if "embeddings" not in data:
            st.error(f"Embedding Error: {data}")
            return []
        return data["embeddings"]
    except Exception as e:
        st.error(f"Failed to connect to local embedding server: {e}")
        return []

def inference(prompt_text):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt_text, "stream": False}
        )
        return r.json()
    except Exception as e:
        st.error(f"Failed to connect to local Llama endpoint: {e}")
        return {"response": "Inference connection error."}

import os
import whisper
import torch

def transcribe_voice_input(audio_bytes):
    """
    Saves the browser's recorded audio bytes to a temporary file,
    runs it through the local Whisper model, and extracts the text.
    """
    # 1. Define the temporary file name
    temp_audio_path = "temp_user_input.wav"
    
    # 2. Write the raw binary audio data from the browser into the file
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)
        
    # 3. Detect if an NVIDIA GPU is available to accelerate transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4. Load the lightweight 'small' model from your dedicated directory
    model = whisper.load_model("base.en", download_root="E:\\whisper_models", device=device)
    
    # 5. Run the transcription loop (forcing fp16=False keeps CPU execution safe if no GPU exists)
    result = model.transcribe(
        audio=temp_audio_path, # Set to "hi" if you want to speak in Hindi
        fp16=False
    )
    
    # 6. Clean up the text by removing trailing whitespace
    clean_text = result["text"].strip()
    print(f"🎤 Whisper Heard: {clean_text}")
    return clean_text
def text_to_speech_autoplay(text):
    """
    Converts text to an MP3 file using gTTS, encodes it to Base64,
    and injects a hidden HTML audio element to trigger instant browser playback.
    """
    # 1. Generate the speech audio file from text
    tts = gTTS(text=text, lang='en', slow=False)
    
    # 2. Save it locally as a temporary asset
    temp_mp3_path = "temp_ai_response.mp3"
    tts.save(temp_mp3_path)
    
    # 3. Read the file back as binary bytes
    with open(temp_mp3_path, "rb") as f:
        audio_bytes = f.read()
        
    # 4. Convert the binary bytes into a Base64 string so it can live inside HTML text
    base64_audio = base64.b64encode(audio_bytes).decode()
    
    # 5. Construct the native HTML audio player with the autoplay attribute enabled
    audio_html = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
        </audio>
        """
    
    # 6. Inject the HTML safely into the Streamlit interface webpage context
    st.markdown(audio_html, unsafe_allow_html=True)
# --- SIDEBAR: DYNAMIC VIDEO ADDITION ---
with st.sidebar:
    st.header("📚 Update Knowledge Base")
    st.write(f"Current Chunks Indexed: `{len(st.session_state.df)}`")
    
    video_url = st.text_input("Paste YouTube Lecture URL:", placeholder="https://www.youtube.com/watch?v=...")
    lecture_num = st.text_input("Assign Video Number:", placeholder="e.g., 45")
    lecture_title = st.text_input("Assign Video Title:", placeholder="e.g., Introduction to CSS Flexbox")
    
    if st.button("Process & Index Video", type="primary"):
        if not video_url or not lecture_num or not lecture_title:
            st.error("Please fill in the URL, video number, and title fields.")
        else:
            # 1. Dynamic Audio Extraction via yt-dlp
            with st.spinner("Extracting audio stream from YouTube link..."):
                os.makedirs("downloads", exist_ok=True)
                audio_filename = f"downloads/audios{lecture_num}_{lecture_title}"
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '128',
                    }],
                    'outtmpl': audio_filename,
                    
                    # --- RESILIENCE SETTINGS ---
                    'retries': 10,                 
                    'fragment_retries': 10,        
                    'http_chunk_size': 10485760,   
                    'nocheckcertificate': True,    
                    'ignoreerrors': True,
                    
                    # --- ADD THIS TO BYPASS THE ANDROID VR BLOCK ---
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['default'] 
                        }
                    }
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    target_audio_path = f"{audio_filename}.mp3"
                    st.success("Audio extracted successfully!")
                except Exception as e:
                    st.error(f"Audio Extraction failed: {e}")
                    target_audio_path = None
            
            # 2. Transcription with Whisper (Preserving your exact arguments)
            if target_audio_path and os.path.exists(target_audio_path):
                with st.spinner("Transcribing audio chunks (Whisper Large-V2)..."):
                    try:
                        # Utilizing your download path logic configuration
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        st.write(f"Running transcription on: `{device.upper()}`")

                        # Pass the device parameter when loading the model
                        model = whisper.load_model("small", download_root="E:\\whisper_models", device=device)
                        result = model.transcribe(
                            audio=target_audio_path,
                            language="hi",
                            task="translate",
                            word_timestamps=False
                        )
                        
                        # Pack segments exactly as you structured them previously
                        new_chunks = []
                        for segment in result["segments"]:
                            new_chunks.append({
                                "number": lecture_num,
                                "title": lecture_title,
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment["text"]
                            })
                        st.success("Transcription complete!")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        new_chunks = None
                
                # 3. Embedding Vectorization & Appending (Your precise DataFrame logic)
                if new_chunks:
                    with st.spinner("Generating BGE-M3 vector embeddings..."):
                        texts_to_embed = [c['text'] for c in new_chunks]
                        embeddings = create_embedding(texts_to_embed)
                        
                        if embeddings:
                            # Safely set tracking IDs continuous from the current data size
                            current_chunk_id = len(st.session_state.df)
                            my_dicts = []
                            
                            for i, chunk in enumerate(new_chunks):
                                chunk['chunk_id'] = current_chunk_id
                                chunk['embedding'] = embeddings[i]
                                current_chunk_id += 1
                                my_dicts.append(chunk)
                            
                            # Convert new entries to a DataFrame and append to current session state
                            new_df_rows = pd.DataFrame.from_records(my_dicts)
                            st.session_state.df = pd.concat([st.session_state.df, new_df_rows], ignore_index=True)
                            
                            # Save back to disk to preserve state permanently
                            joblib.dump(st.session_state.df, EMBEDDINGS_FILE)
                            st.success(f"Successfully added {len(new_chunks)} chunks to vector index!")
                            st.balloons()

# --- MAIN CHAT SCREEN INTERFACE ---
st.subheader("Interactive Classroom Chat")

# Render persistent historical log elements smoothly
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lock conversation UI execution strictly until context entries are ready inside the DataFrame
if len(st.session_state.df) == 0:
    st.info("Your knowledge base is empty. Please process a video to populate the vector matrix.")
else:
    # 1. Create a structural layout: small column for mic, large column space for clear UI flow
    col1, col2 = st.columns([1, 10])
    
    with col1:
        # Renders the microphone element. Clicking it activates recording; clicking again stops it.
        audio_bytes = audio_recorder(text="", icon_size="2x", icon_name="microphone", key="mic_widget")
    
    # 2. Initialize an empty variable to hold the final question text
    incoming_query = None
    
    # Scenario A: The user utilized the microphone widget
    if audio_bytes:
        with st.spinner("Processing voice input..."):
            incoming_query = transcribe_voice_input(audio_bytes)
    
    # Scenario B: The user typed inside the standard chat bar input frame
    text_input = st.chat_input("Ask a question about the course materials...")
    if text_input:
        incoming_query = text_input

    # 3. RAG Pipeline Trigger (Runs only if a query was captured via voice OR text)
    if incoming_query:
        # Log the user's question into the historical visual state
        st.session_state.messages.append({"role": "user", "content": incoming_query})
        with st.chat_message("user"):
            st.markdown(incoming_query)
        
        # Open assistant response visual frame block
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            with st.spinner("Searching transcripts for matching concepts..."):
                # --- YOUR EXACT RETRIEVAL MATHEMATICS ---
                question_embedding = create_embedding([incoming_query])[0]
                
                similarities = cosine_similarity(
                    np.vstack(st.session_state.df['embedding'].values), 
                    [question_embedding]
                ).flatten()
                
                top_results = min(5, len(similarities))
                max_idx = similarities.argsort()[::-1][0:top_results]
                new_df = st.session_state.df.iloc[max_idx]
                
                # --- PROMPT TEMPLATE INJECTION & OLLAMA INFERENCE ---
                prompt = f"""You are an AI Teaching Assistant for the Sigma Web Development Course.

Your job is to help students find exactly where a topic is taught in the course videos.

=========================
COURSE CONTENT
=========================

{new_df[["title", "number", "start", "end", "text"]].to_json()}

=========================
STUDENT QUESTION
=========================

{incoming_query}

=========================
INSTRUCTIONS
=========================

1. Analyze the course content carefully.
2. Answer ONLY using the provided course content.
3. Mention:
   - Video Number
   - Video Title
   - Relevant Timestamp(s)
4. Explain briefly what is taught in that section.
5. Guide the student to watch the most relevant video first.
6. If information is spread across multiple videos, mention all relevant videos in order.
7. If the answer is not available in the provided content, respond:
   "I could not find enough information in the course material to answer this question."
8. If the question is unrelated to web development or the course, respond:
   "I can only answer questions related to the Sigma Web Development Course."

=========================
RESPONSE FORMAT
=========================

Answer naturally like a teacher.

Example:

You can find this topic in Video 45: CSS Flexbox Basics.

Relevant timestamps:
• 12:30 - 18:45
• 20:10 - 24:50

In this section, Flexbox properties such as justify-content and align-items are explained with examples.

I recommend watching Video 45 first and then Video 46 for advanced layouts.

Now answer the student's question.
"""
                
                # Run your local Llama 3.2 engine
                inference_output = inference(prompt)
                response_text = inference_output.get('response', 'Failed to generate response.')
                
                # 4. Display the text answer in the chat bubble
                response_container.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # 5. Trigger the text-to-speech audio loop to read the answer out loud
                text_to_speech_autoplay(response_text)