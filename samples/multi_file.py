import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

def prepare_30s_chunk(file_path):
    audio = AudioSegment.from_wav(file_path)
    target_duration = 30 * 1000  # 30 seconds in milliseconds
    
    if len(audio) < target_duration:
        # Add silence if the audio is too short
        silence = AudioSegment.silent(duration=target_duration - len(audio))
        audio = audio + silence
    elif len(audio) > target_duration:
        # Cut the audio if it's too long
        audio = audio[:target_duration]
    
    # Save the processed chunk to a temporary file
    temp_file = f"temp_{os.path.basename(file_path)}"
    audio.export(temp_file, format="wav")
    return temp_file

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Process multiple chunks
chunk_files = ["./chunk1.wav", "./chunk2.wav"]
chunk_start_times = [0, 30]  # Adjusted to 30 seconds intervals
processed_chunks = [prepare_30s_chunk(file) for file in chunk_files]
chunk_results = [pipeline.process_chunk(file) for file in processed_chunks]

# Clean up temporary files
for temp_file in processed_chunks:
    os.remove(temp_file)

# Access local diarization and overlapping speech for each chunk
for i, chunk_result in enumerate(chunk_results):
    if chunk_result is not None:
        print(f"\nLocal diarization for chunk {i}:")
        print(chunk_result['local_diarization'])
        
        print(f"Overlapping speech in chunk {i}:")
        if chunk_result['overlapping_speech']:
            for start, end in chunk_result['overlapping_speech']:
                print(f"  {start:.2f}s - {end:.2f}s")
        else:
            print("  No overlapping speech detected")

# Perform global diarization
global_diarization, centroids = pipeline.global_diarization(
    chunk_results,
    num_speakers=None,
    min_speakers=1,
    max_speakers=10
)

# Print the global diarization results
print("\nGlobal diarization results:")
for turn, _, speaker in global_diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")

# Print speaker centroids
print("\nSpeaker centroids:")
for i, centroid in enumerate(centroids):
    print(f"SPEAKER_{i:02d}: {centroid[:5]}...")  # Print first 5 values of each centroid