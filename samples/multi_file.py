from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# apply pretrained pipeline
diarization = pipeline("./chunk3_extra.wav")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")