import subprocess
from pathlib import Path
import soundfile as sf
import dfpwm
import resampy
import numpy as np
import os
import math

# --- Configuration ---
MAX_SPLIT_SIZE_KB = 480  # Maximum file size for each split part in Kilobytes
MAX_SPLIT_SIZE_BYTES = MAX_SPLIT_SIZE_KB * 1024

# Global variables for cleanup
keep = False
mp3_file = None
wav_file = None
# --- End Configuration ---


def run_ffmpeg(command):
    """Run ffmpeg and raise a clear error if it fails."""
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Make sure ffmpeg is installed and in your PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed with error code {e.returncode}")


def convert_flac_to_mp3(input_path, output_path, bitrate='128k'):
    command = ['ffmpeg', '-y', '-i', str(input_path), '-b:a', bitrate, str(output_path)]
    run_ffmpeg(command)
    print(f"Converted {input_path.name} → {output_path.name}")


def convert_mp3_to_wav(mp3_path, wav_path):
    print(f"Converting MP3 → WAV: {mp3_path.name} → {wav_path.name}")
    # Force 32768Hz sample rate, 1 channel (mono), and 16-bit depth (pcm_s16le)
    command = [
        'ffmpeg',
        '-y',
        '-i', str(mp3_path),
        '-ar', '32768',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        str(wav_path),
    ]
    run_ffmpeg(command)
    print(f"Converted {mp3_path.name} → {wav_path.name} (32768Hz, Mono, 16-bit)")


def encode_audio_to_dfpwm(input_file, output_file):
    print(f"Encoding WAV → DFPWM: {input_file.name} → {output_file.name}")

    try:
        data, sample_rate = sf.read(str(input_file), always_2d=True)

        # data shape: (n_samples, n_channels) → take mono
        if data.shape[1] > 1:
            # average channels to mono
            data = data.mean(axis=1)
        else:
            data = data[:, 0]

        # Ensure we have a 1D numpy array
        data = np.asarray(data, dtype=np.float32)

        # Target sample rate from dfpwm module or fallback to 32768
        target_rate = getattr(dfpwm, "SAMPLE_RATE", 32768)

        if sample_rate != target_rate:
            print(f"Resampling from {sample_rate} Hz to {target_rate} Hz using resampy...")
            data = resampy.resample(data, sample_rate, target_rate)

        # Convert float [-1.0, 1.0] → int16
        data = np.clip(data, -1.0, 1.0)
        data_int16 = (data * 32767).astype(np.int16)

        # Encode with dfpwm (handle a couple of common APIs)
        if hasattr(dfpwm, "compress"):
            dfpwm_bytes = dfpwm.compress(data_int16)
        elif hasattr(dfpwm, "compressor"):
            dfpwm_bytes = dfpwm.compressor(data_int16)
        else:
            raise RuntimeError(
                "Your dfpwm module does not expose 'compress' or 'compressor'. "
                "Check its documentation and adjust this function accordingly."
            )

        output_file.write_bytes(dfpwm_bytes)
        print(f"Encoded to {output_file.name} ({len(dfpwm_bytes) / 1024:.2f} KB)")

        # Cleanup intermediate files if requested
        global mp3_file, wav_file, keep
        if not keep:
            if mp3_file and mp3_file.exists():
                mp3_file.unlink()
            if wav_file and wav_file.exists() and wav_file != input_file:
                wav_file.unlink()
            print("Deleted intermediate files (.mp3 & .wav)")

        return len(dfpwm_bytes)  # Return the size of the full DFPWM file

    except Exception as e:
        print("Error during DFPWM encoding:", e)
        return 0


def split_dfpwm(input_file, output_prefix_path, parts=8):
    """
    input_file: Path to the full dfpwm file
    output_prefix_path: Path WITHOUT extension, e.g. script_dir / stem
    """
    try:
        data = input_file.read_bytes()
        total_size = len(data)
        if parts <= 0:
            raise ValueError("Number of parts must be at least 1.")

        part_size = total_size // parts
        remainder = total_size % parts  # remaining bytes to distribute

        print(f"\nSplitting {input_file.name} (Total size: {total_size} bytes) into {parts} parts...")

        offset = 0
        for i in range(parts):
            # Distribute remainder bytes: one extra byte to first 'remainder' parts
            extra = 1 if i < remainder else 0
            current_part_size = part_size + extra

            start = offset
            end = start + current_part_size
            offset = end

            part_data = data[start:end]
            output_file = output_prefix_path.parent / f"{output_prefix_path.name}_part{i+1}.dfpwm"
            output_file.write_bytes(part_data)
            print(f"Saved {output_file.name} ({len(part_data) / 1024:.2f} KB)")

        # Remove the original large dfpwm file after splitting
        input_file.unlink()
    except Exception as e:
        print("Error during DFPWM splitting:", e)


def main():
    global keep, mp3_file, wav_file

    # Figure out the script directory (where this .py file is)
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (e.g. interactive)
        script_dir = Path.cwd()

    # Create input folder inside script folder
    input_dir = script_dir / "input_audio"
    input_dir.mkdir(exist_ok=True)

    print(f"Input folder created/used: {input_dir}")
    print("Put your .flac, .mp3, or .wav files in this folder, then run the script.")
    print()

    # Collect all supported files in the input folder
    audio_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".flac", ".mp3", ".wav")
    ]

    if not audio_files:
        print("No audio files found in 'input_audio'.")
        print("Add some files and run this script again.")
        return

    print("Found the following files to convert:")
    for f in audio_files:
        print(" •", f.name)
    print()

    # Ask once whether to keep .wav/.mp3
    print("If you want to keep .wav and .mp3, type anything and press Enter.")
    print("If you DON'T want to keep them, just press Enter without typing.")
    keep_input = input(": ").strip()
    keep = bool(keep_input)

    # Ask once for split behavior (applied to ALL files)
    print("\n--- Split Options (applied to EVERY file) ---")
    print("1) Use a specific number of parts for all files.")
    print(f"2) Auto-select parts for each file to keep each part under {MAX_SPLIT_SIZE_KB} KB.")

    split_choice = input("Enter 1 or 2: ").strip()

    parts_fixed = None
    if split_choice == "1":
        parts_fixed = int(input("How many parts to split each file into: ").strip())
        if parts_fixed <= 0:
            raise ValueError("Number of parts must be 1 or greater.")
    elif split_choice == "2":
        print("Auto-splitting each file based on size limit.")
    else:
        raise ValueError("Invalid choice. Please enter 1 or 2.")

    # Process each file in the input folder
    for file in audio_files:
        print("\n==============================")
        print(f"Processing: {file.name}")
        print("==============================")

        stem = file.stem

        # intermediate files stored in same input folder
        mp3_file = file.with_suffix(".mp3")
        wav_file = file.with_suffix(".wav")

        # final dfpwm (before splitting) stored next to the script
        dfpwm_file = script_dir / f"{stem}.dfpwm"
        dfpwm_prefix = script_dir / stem  # for split output names

        # --- STEP 1: Conversion to WAV ---
        if file.suffix.lower() == ".flac":
            convert_flac_to_mp3(file, mp3_file)
            convert_mp3_to_wav(mp3_file, wav_file)
        elif file.suffix.lower() == ".mp3":
            mp3_file = file
            convert_mp3_to_wav(mp3_file, wav_file)
        elif file.suffix.lower() == ".wav":
            wav_file = file
        else:
            print(f"Skipping unsupported file type: {file.name}")
            continue

        # --- STEP 2: Encoding to DFPWM ---
        full_dfpwm_size = encode_audio_to_dfpwm(wav_file, dfpwm_file)
        if full_dfpwm_size == 0:
            print(f"Skipping splitting for {file.name} due to encoding failure.")
            continue

        # --- STEP 3: Determine number of parts ---
        if split_choice == "1":
            parts_to_split = parts_fixed
        else:
            parts_to_split = max(
                1,
                math.ceil(full_dfpwm_size / MAX_SPLIT_SIZE_BYTES)
            )
            print(
                f"Auto-selected {parts_to_split} parts for {file.name} "
                f"to meet the {MAX_SPLIT_SIZE_KB} KB per-part limit."
            )

        # --- STEP 4: Split and Cleanup ---
        split_dfpwm(dfpwm_file, dfpwm_prefix, parts_to_split)

    print("\n✅ All files processed!")


if __name__ == "__main__":
    main()

