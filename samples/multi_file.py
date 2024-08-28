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
    return temp_file, audio

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

def process_local_diarization(chunk_result, chunk_index, audio_segment, chunk_duration=30):
    local_diarization = chunk_result['local_diarization']
    overlapping_speech = chunk_result['overlapping_speech']
    
    # Log original local diarization
    print(f"\nOriginal local diarization for chunk {chunk_index}:")
    for turn, _, speaker in local_diarization.itertracks(yield_label=True):
        print(f"  {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
    
    # Log overlapping speech
    print(f"\nOverlapping speech in chunk {chunk_index}:")
    if overlapping_speech:
        for start, end in overlapping_speech:
            print(f"  {start:.2f}s - {end:.2f}s")
    else:
        print("  No overlapping speech detected")
    
    # Group same speaker turns with less than 2s pause
    grouped_turns = group_speaker_turns(local_diarization, max_pause=2)
    
    # Handle overlapping speech
    grouped_turns = handle_overlapping_speech(grouped_turns, overlapping_speech)
    
    # Remove speaker turns less than 1s
    grouped_turns = remove_short_turns(grouped_turns, min_duration=1)
    
    # Create split files and overview dict
    split_files, overview = create_split_files(grouped_turns, chunk_index, audio_segment, chunk_duration)
    
    return split_files, overview

def group_speaker_turns(diarization, max_pause=2):
    grouped_turns = []
    current_group = None
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if current_group is None or speaker != current_group['speaker'] or turn.start - current_group['end'] > max_pause:
            if current_group is not None:
                grouped_turns.append(current_group)
            current_group = {'speaker': speaker, 'start': turn.start, 'end': turn.end}
        else:
            current_group['end'] = turn.end
    
    if current_group is not None:
        grouped_turns.append(current_group)
    
    return grouped_turns

def handle_overlapping_speech(grouped_turns, overlapping_speech):
    new_grouped_turns = sorted(grouped_turns, key=lambda x: x['start'])
    
    for i in range(len(new_grouped_turns) - 1):
        current_turn = new_grouped_turns[i]
        next_turn = new_grouped_turns[i + 1]
        
        if current_turn['end'] > next_turn['start']:
            # There's an overlap
            overlap_duration = current_turn['end'] - next_turn['start']
            current_duration = current_turn['end'] - current_turn['start']
            next_duration = next_turn['end'] - next_turn['start']
            
            if current_duration > next_duration:
                # Prioritize the longer segment (current_turn)
                next_turn['start'] = current_turn['end']
            else:
                # Prioritize the longer segment (next_turn)
                current_turn['end'] = next_turn['start']
    
    # Now handle standalone overlapping speech
    for overlap_start, overlap_end in overlapping_speech:
        is_standalone = all(
            overlap_end <= turn['start'] or overlap_start >= turn['end']
            for turn in new_grouped_turns
        )
        
        if is_standalone:
            prev_turn = next((t for t in reversed(new_grouped_turns) if t['end'] <= overlap_start), None)
            next_turn = next((t for t in new_grouped_turns if t['start'] >= overlap_end), None)
            
            if prev_turn and overlap_start - prev_turn['end'] <= 2:
                prev_turn['end'] = overlap_end
            elif next_turn and next_turn['start'] - overlap_end <= 2:
                next_turn['start'] = overlap_start
    
    return [turn for turn in new_grouped_turns if turn['end'] > turn['start']]

def remove_short_turns(grouped_turns, min_duration=1):
    return [turn for turn in grouped_turns if turn['end'] - turn['start'] >= min_duration]

def create_split_files(grouped_turns, chunk_index, audio_segment, chunk_duration):
    split_files = []
    overview = []
    
    for turn_index, turn in enumerate(grouped_turns):
        start_time = turn['start']
        end_time = turn['end']
        speaker = turn['speaker']
        
        file_name = f"chunk_{chunk_index}_turn_{turn_index:02d}_speaker_{speaker}_{start_time:.2f}_{end_time:.2f}.wav"
        
        # Extract audio segment
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        turn_audio = audio_segment[start_ms:end_ms]
        
        # Export audio segment
        turn_audio.export(file_name, format="wav")
        
        split_files.append(file_name)
        overview.append({
            'chunk_index': chunk_index,
            'turn_index': turn_index,
            'speaker': speaker,
            'start_time': start_time,
            'end_time': end_time,
            'file_name': file_name
        })
    
    return split_files, overview


# Process multiple chunks
chunk_files = ["./overlapping.wav"]
chunk_start_times = [0, 30]  # Adjusted to 30 seconds intervals
processed_chunks = [prepare_30s_chunk(file) for file in chunk_files]
chunk_results = [pipeline.process_chunk(temp_file) for temp_file, _ in processed_chunks]

all_split_files = []
all_overviews = []

# Process local diarization for each chunk
for i, ((temp_file, audio_segment), chunk_result) in enumerate(zip(processed_chunks, chunk_results)):
    if chunk_result is not None:
        split_files, overview = process_local_diarization(chunk_result, i, audio_segment)
        all_split_files.extend(split_files)
        all_overviews.extend(overview)

# Clean up temporary files
for temp_file, _ in processed_chunks:
    os.remove(temp_file)

# Print the overview of processed split files
print("\nOverview of processed split files:")
for item in all_overviews:
    print(f"Chunk {item['chunk_index']}, Turn {item['turn_index']}, {item['speaker']}: {item['start_time']:.2f}s - {item['end_time']:.2f}s, File: {item['file_name']}")

# Perform global diarization (if needed)
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