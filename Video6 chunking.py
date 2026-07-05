import json
import os
import whisper

# 1. Load the model
model = whisper.load_model("large-v2", download_root="E:\\whisper_models")

# 2. Define the specific remaining file you want to process
target_audio = "audios6_SEO and Core Web Vitals in HTML.mp3"

# Ensure the jsons directory exists so Python doesn't throw an error
os.makedirs("jsons", exist_ok=True)

if "_" in target_audio:
    # Extract metadata from the filename
    number = target_audio.split("_")[0]  # Extracts "audios6"
    title = target_audio.split("_")[1][:-4]  # Extracts "SEO and Core Web Vitals in HTML"
    print(f"Processing: {number} - {title}")

    # 3. Transcribe the specific file dynamically
    result = model.transcribe(
        audio=f"audios/{target_audio}",
        language="hi",
        task="translate",
        word_timestamps=False,
    )

    # 4. Extract segments
    chunks = []
    for segment in result["segments"]:
        chunks.append(
            {
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
        )
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    # 5. Save dynamically into the 'jsons' folder
    output_json_path = f"jsons/{number}_{title}.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=4)

    print(f"Successfully dumped chunks to {output_json_path}")