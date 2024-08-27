import numpy as np
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Process multiple chunks
chunk_files = ["./overlapping.wav"]
chunk_results = [pipeline.process_chunk(file) for file in chunk_files]

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
diarization, centroids = pipeline.global_diarization(chunk_results)

# Print the global diarization results
print("\nGlobal diarization results:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

print("\nSpeaker centroids:")
for i, centroid in enumerate(centroids):
    print(f"Speaker {i}: {centroid[:5]}...")  # Print first 5 values of each centroid