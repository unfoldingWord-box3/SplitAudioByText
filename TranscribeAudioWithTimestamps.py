import whisper
import os
import ssl
import json
from pydub import AudioSegment
from whisper import Whisper

model_name = "large-v3" # Options: ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo']
output_dir = "extracted_segments"

def transcribe_audio_to_text_with_timestamps(model:Whisper, file_path: str):
    """
    Transcribes an audio file to time-stamped text using the Whisper model.
    
    Args:
        file_path (str): Path to the audio file to be transcribed.
    
    Returns:
        list: A list of dictionaries containing start time, end time, and the transcribed text.
    """

    # Transcribe the audio with timestamps
    print(f"Transcribing audio file: {file_path}")
    result = model.transcribe(file_path, task="transcribe", verbose=True, word_timestamps=True)

    # Extract segments with timestamps
    segments = result.get("segments", [])

    transcription = []
    for segment in segments:
        transcription.append({
            "start": segment["start"],  # Start time in seconds
            "end": segment["end"],  # End time in seconds
            "text": segment["text"]  # Transcribed text
        })

    return transcription


def extractSegmentsToAudioFiles(audio_file, timestamps):
    global output_dir
    
    # Get file name without path and extension
    audio_base_name=os.path.splitext(os.path.basename(audio_file))[0]
    # Extract audio segments
    audio = AudioSegment.from_file(audio_file)

    for i, segment in enumerate(timestamps):
        start_ms = int(segment['start'] * 1000)  # Convert to milliseconds
        end_ms = int(segment['end'] * 1000)  # Convert to milliseconds
        segment_audio = audio[start_ms:end_ms]
        segment_file = os.path.join(output_dir, f"{audio_base_name}_{start_ms:06d}_{end_ms:06d}.mp3")
        segment_audio.export(segment_file, format="mp3")
        print(f"Exported segment {i + 1} to {segment_file}")



def splitAudioFile(model:Whisper, audio_file):
    global output_dir
    
    try:
        # Perform transcription
        timestamps = transcribe_audio_to_text_with_timestamps(model, audio_file)
    
        # Save the transcription to a JSON file
        audio_base_name=os.path.basename(audio_file)
        json_file = f"./{output_dir}/{audio_base_name}.timestamps.json"
        with open(json_file, "w") as f:
            json.dump(timestamps, f, indent=4)

        print(f"Transcription saved to {json_file}")

        extractSegmentsToAudioFiles(audio_file, timestamps)

        # # Print the transcription in a readable format
        # print("\n--- Transcription with Timestamps ---\n")
        # for segment in timestamps:
        #     print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s]: {segment['text']}")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    global model_name, output_dir
    
    ssl._create_default_https_context = ssl._create_unverified_context

    print(f"Loading the Model: {model_name}")
    model = whisper.load_model("%s" % model_name)

    os.makedirs(output_dir, exist_ok=True)
    
    audio_folder = "./audio"
    for file_name in sorted(os.listdir(audio_folder)):
        audio_file = os.path.join(audio_folder, file_name)
        if os.path.isfile(audio_file):
            print(f"Splitting file: {audio_file}")
            splitAudioFile(model, audio_file)


if __name__ == "__main__":
    
    main()
