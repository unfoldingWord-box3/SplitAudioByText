import whisper
import os
import ssl


def transcribe_audio_to_text_with_timestamps(file_path: str):
    """
    Transcribes an audio file to time-stamped text using the Whisper model.
    
    Args:
        file_path (str): Path to the audio file to be transcribed.
    
    Returns:
        list: A list of dictionaries containing start time, end time, and the transcribed text.
    """

    ssl._create_default_https_context = ssl._create_unverified_context

    # Load the Whisper model
    model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"

    # Transcribe the audio with timestamps
    print(f"Transcribing audio file: {file_path}")
    result = model.transcribe(file_path, task="transcribe", verbose=True)

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


if __name__ == "__main__":
    audio_file = "./audio/en_obs_01_128kbps.mp3"  # Convert to absolute path
    try:
        import json

        # Perform transcription
        timestamps = transcribe_audio_to_text_with_timestamps(audio_file)

        # Save the transcription to a JSON file
        json_file = "transcription_with_timestamps.json"
        with open(json_file, "w") as f:
            json.dump(timestamps, f, indent=4)

        print(f"Transcription saved to {json_file}")

        # Print the transcription in a readable format
        print("\n--- Transcription with Timestamps ---\n")
        for segment in timestamps:
            print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s]: {segment['text']}")

    except Exception as e:
        print(f"An error occurred: {e}")
