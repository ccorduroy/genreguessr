# checks that all .wav files are valid and quarantines corrupted/invalid downloads to another folder outside dataset.

import os
import wave
import shutil

def is_valid_wav(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            wav_file.getparams()
        return True
    except wave.Error:
        return False
    except Exception:
        return False

def scan_and_quarantine_wavs(root_folder, quarantine_folder="quarantined_data"):
    bad_files = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(dirpath, filename)
                if not is_valid_wav(file_path):
                    print(f"[!] Invalid or corrupted: {file_path}")
                    bad_files.append(file_path)

                    # Compute relative path to preserve folder structure
                    rel_path = os.path.relpath(file_path, root_folder)
                    quarantine_path = os.path.join(quarantine_folder, rel_path)

                    # Create quarantine subfolders if needed
                    os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)

                    # Move the bad file
                    shutil.move(file_path, quarantine_path)
                    print(f"    → Moved to: {quarantine_path}")

    if bad_files:
        print(f"\nMoved {len(bad_files)} invalid .wav file(s) to quarantine: '{quarantine_folder}'.")
    else:
        print("\nAll .wav files are valid")

    return bad_files

# Example usage:
scan_and_quarantine_wavs("./gtzan/genres_original")
