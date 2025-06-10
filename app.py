

!pip install -q transformers datasets pydub torchaudio accelerate

import os
import torch
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

os.environ["HF_TOKEN"] = ""



# ========== Step 1: Split audio into 1-minute chunks ==========

def split_audio(file_path, chunk_length_ms=60_000):  # 1 minute
    audio = AudioSegment.from_file(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        paths.append(chunk_path)
    return paths

# ========== Step 2: Load Whisper-Medium and Processor ==========

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
model.config.forced_decoder_ids = None  # Avoid forcing language tokens

#  ========== Step 3: Transcribe Each Chunk ==========

#  ========== Step 3: Transcribe Each Chunk ==========

def transcribe_chunk(file_path):
    import torchaudio
    from torchaudio.transforms import Resample

    speech_array, sr = torchaudio.load(file_path)

    # Resample to 16000 Hz if necessary
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        speech_array = resampler(speech_array)
        sr = 16000

    input_features = processor(
        speech_array.squeeze().numpy(),
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features.to(device)

    # Use generate_with_forced_decoder_ids to correctly force the language
    predicted_ids = model.generate(input_features, max_new_tokens=225, language="en")

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# ========== Step 4: Combine Chunks ==========

def transcribe_all_chunks(chunk_paths):
    full_text = ""
    for i, path in enumerate(chunk_paths):
        print(f"⏱️ Transcribing chunk {i+1}/{len(chunk_paths)}")
        text = transcribe_chunk(path)  # ⬅️ this is where the single chunk function is used
        full_text += text + "\n"
    return full_text

from transformers import pipeline as hf_pipeline
import os

def summarize_text(text, chunk_size=1000):
    # Explicitly set the device to 'cpu' for the summarization model
    summarizer = hf_pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum",
        token=os.environ["HF_TOKEN"],
        device="cpu"  # Use CPU for summarization
    )

    # Split transcript into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"🧠 Summarizing chunk {i+1}/{len(chunks)}")
        try:
            summary = summarizer(chunk, max_length=200, min_length=30, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Error in chunk {i+1}: {e}")
            continue

    return "\n".join(summaries)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # Explicitly import torch

# Load Phi-2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}") # Added for clarity

try:
    # Try loading with float16 first
    phi_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16 # Explicitly set dtype
    ).to(device)
except Exception as e:
    print(f"Failed to load Phi-2 with float16: {e}")
    print("Trying to load with float32...")
    # If float16 fails, try loading with float32
    try:
        phi_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32 # Explicitly set dtype
        ).to(device)
    except Exception as e_float32:
        print(f"Failed to load Phi-2 with float32: {e_float32}")
        print("Could not load Phi-2 model on device. Please check your GPU and PyTorch installation.")
        # Handle the case where the model cannot be loaded
        phi_model = None # Or exit, or set to CPU

# Load tokenizer (usually works fine on CPU)
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Ensure functions handle potentially failed model loading
def extract_bullets(summary):
    if phi_model is None:
        print("Phi-2 model not loaded. Cannot extract bullets.")
        return "Error: Model not available."

    prompt = f"""Extract the key discussion points as clear bullet points from this meeting summary:\n\n{summary}\n\nBullet Points:"""
    # Ensure inputs are on the correct device
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    result = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Bullet Points:")[-1].strip()

def extract_action_items(summary):
    if phi_model is None:
        print("Phi-2 model not loaded. Cannot extract action items.")
        return "Error: Model not available."

    prompt = f"""From this meeting summary, extract actionable items with responsible persons and deadlines if mentioned.\n\n{summary}\n\nAction Items:"""
    # Ensure inputs are on the correct device
    inputs = phi_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    result = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Action Items:")[-1].strip()

# def extract_action_items(summary):
#     prompt = f"""From this meeting summary, extract actionable items with responsible persons and deadlines if mentioned.\n\n{summary}\n\nAction Items:"""
#     inputs = phi_tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = phi_model.generate(**inputs, max_new_tokens=300)
#     result = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return result.split("Action Items:")[-1].strip()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def extract_meeting_notes(text):
    prompt = f"""
Given the following meeting transcript, extract:

1. A short summary.
2. Key discussion points in bullet format.
3. Action items with responsible people and deadlines if any.

Transcript:
{text}
"""
    result = pipe(prompt, max_new_tokens=512)[0]["generated_text"]
    return result

# Example usage:
notes = extract_meeting_notes(full_transcript)
print(notes)

# ========== RUN ALL ==========

# Upload your long meeting audio file to Colab (e.g., "/content/meeting.mp3")
audio_path = "meet.mp3"  # CHANGE THIS TO YOUR FILE
chunk_paths = split_audio(audio_path)



# Summarize it

# Transcribe all audio chunks
full_transcript = transcribe_all_chunks(chunk_paths)
print("\n📝 Full Transcript Preview:\n", full_transcript[:1000])

summary = summarize_text(full_transcript)
print("\n📄 Summary:\n", summary)

bullet_points = extract_bullets(summary)
print("\n🔹 Key Discussion Points:\n", bullet_points)

actions = extract_action_items(summary)
print("\n✅ Action Items:\n", actions)

