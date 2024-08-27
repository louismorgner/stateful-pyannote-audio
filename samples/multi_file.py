import numpy as np
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Process multiple chunks
chunk_files = ["./chunk3_extra.wav"]
chunk_results = [pipeline.process_chunk(file) for file in chunk_files]

# Perform global diarization
diarization, centroids = pipeline.global_diarization(chunk_results)

# Print the results
print("Global diarization results:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

print("\nSpeaker centroids:")
for i, centroid in enumerate(centroids):
    print(f"Speaker {i}: {centroid[:5]}...")  # Print first 5 values of each centroid