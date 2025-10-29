import os
import ffmpeg
import time
import re
import multiprocessing
import traceback
import shutil 
from pydub import AudioSegment, silence 

# --- Library Import Checks ---
whisper = None; GoogleTranslator = None; gTTS = None
Separator = None; torch = None
DEVICE = "cpu"

try:
    import torch; print("DEBUG: PyTorch imported successfully."); DEVICE = "cpu"; print(f"DEBUG: Forcing device to: {DEVICE}")
except Exception: print("FATAL ERROR: PyTorch import failed.")

try:
    import whisper; print("DEBUG: Whisper imported successfully.")
except Exception as e: print(f"FATAL ERROR importing whisper: {e}")

try:
    from deep_translator import GoogleTranslator; print("DEBUG: Deep Translator imported successfully.")
except Exception as e: print(f"FATAL ERROR importing deep_translator: {e}")

try:
    from gtts import gTTS; print("DEBUG: gTTS imported successfully.")
except Exception as e: print(f"FATAL ERROR importing gtts: {e}")

try:
    if 'AudioSegment' not in globals():
        from pydub import AudioSegment; 
    print("DEBUG: Pydub imported successfully.")
except Exception as e: print(f"FATAL ERROR importing pydub: {e}")

try:
    from spleeter.separator import Separator; print("DEBUG: Spleeter Separator imported successfully.")
except Exception as e: print(f"FATAL ERROR importing spleeter: {e}")
# -----------------------------------------------------------


# --- File Names & Settings ---
DEFAULT_SPLEETER_OUTPUT_FOLDER = "spleeter_output"
DEFAULT_SUBTITLE_FILE = "subtitles.srt"
DEFAULT_FINAL_VIDEO_FILE = "dubbed_video.mp4"

# --- Helper Functions ---
def time_str_to_ms(time_str):
    """Converts a time string (e.g., '123.45') to milliseconds."""
    try: return int(float(time_str) * 1000)
    except Exception: return 0

def format_time_srt(seconds_float):
    """Formats seconds (float) into SRT time format HH:MM:SS,ms."""
    try:
        if not isinstance(seconds_float, (int, float)) or seconds_float < 0: seconds_float = 0
        total_seconds = int(seconds_float); milliseconds = int((seconds_float - total_seconds) * 1000)
        hours, remainder = divmod(total_seconds, 3600); minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    except Exception: return "00:00:00,000"


# --- NEW FUNCTION: Clean Transcription Timing (Step 2.5) ---
def clean_transcription_timing(transcription_path, output_path, min_gap_s=0.1):
    print(f"\n--- Entering Step 2.5: Clean Timing & Remove Overlap (Min Gap: {min_gap_s}s) ---")
    if not os.path.exists(transcription_path): 
        print(f"❌ FAIL Step 2.5: Input transcription missing."); return False
    
    try:
        if os.path.exists(output_path): print(f"'{output_path}' exists. Skipping."); return True
        
        with open(transcription_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        cleaned_lines = []
        previous_end_time = 0.0 # Track the end time of the previous segment
        
        for i, line in enumerate(lines):
            line = line.strip()
            match = re.match(r'\[(\d+\.\d+) --> (\d+\.\d+)] (.*)', line)
            
            if not match:
                cleaned_lines.append(line)
                continue
            
            start_f = float(match.group(1))
            end_f = float(match.group(2))
            text = match.group(3).strip()
            
            # 1. Check for Overlap/Insufficient Gap
            required_start_time = previous_end_time + min_gap_s
            
            if start_f < required_start_time:
                new_start_f = required_start_time
                
                # Check if the segment duration is still meaningful (e.g., > 0.1s)
                if (end_f - new_start_f) > 0.1 and text: 
                    start_f = new_start_f
                else:
                    continue
            
            # 2. Update the previous_end_time
            previous_end_time = end_f

            # 3. Save the cleaned line
            cleaned_lines.append(f"[{start_f:.2f} --> {end_f:.2f}] {text}")

        # Save the cleaned transcription to a new file
        with open(output_path, "w", encoding="utf-8") as f:
            for line in cleaned_lines:
                f.write(line + "\n")
        
        print(f"Timing cleanup saved to '{output_path}'")
        return True
        
    except Exception as e: 
        print(f"❌ FAIL Step 2.5 (Timing Cleanup): {e}"); traceback.print_exc(); return False


# --- FUNCTION 1: Extract Audio ---
def extract_audio(video_path, output_audio_path):
    print(f"\n--- Entering Step 1: Extract Audio ---")
    print(f"  Video Path: {video_path}\n  Output Path: {output_audio_path}")
    try:
        if not os.path.exists(video_path): print(f"❌ FAIL Step 1: Input video file not found: {video_path}"); return False
        if os.path.exists(output_audio_path): print(f"'{output_audio_path}' exists. Skipping extraction."); return True
        stream = ffmpeg.input(video_path); stream_audio = stream.audio
        stream = ffmpeg.output(stream_audio, output_audio_path, acodec='mp3'); print("  Running ffmpeg to extract audio..."); ffmpeg_cmd = ['ffmpeg', '-loglevel', 'warning']
        ffmpeg.run(stream, overwrite_output=True, quiet=False, cmd=ffmpeg_cmd)
        if not os.path.exists(output_audio_path): print(f"❌ FAIL Step 1: Output audio not created."); return False
        print(f"Audio saved to '{output_audio_path}'"); return True
    except ffmpeg.Error as e:
        print(f"❌ FAIL Step 1 (ffmpeg error):"); stderr = ""
        if e.stderr: 
            try: stderr = e.stderr.decode('utf-8', errors='ignore')
            except: stderr="Could not decode stderr."
        print(stderr); return False
    except Exception as e: print(f"❌ FAIL Step 1 (general error): {e}"); traceback.print_exc(); return False


# --- FUNCTION 2: Transcribe Audio ---
def transcribe_audio(audio_path, output_transcription_path, model_name="base"):
    print(f"\n--- Entering Step 2: Transcribe Audio ---")
    print(f"  Model: {model_name}")
    if whisper is None: print(f"❌ FAIL Step 2: Whisper library missing."); return False
    global DEVICE; print(f"  Device: {DEVICE}")
    if not os.path.exists(audio_path): print(f"❌ FAIL Step 2: Input audio not found: {audio_path}"); return False
    try:
        if os.path.exists(output_transcription_path): print(f"'{output_transcription_path}' exists. Skipping."); return True
        print("  DEBUG: Loading Whisper model..."); load_start = time.time(); model = whisper.load_model(model_name, device=DEVICE); load_end = time.time(); print(f"  DEBUG: Model loaded in {load_end - load_start:.2f}s.")
        print("  DEBUG: Starting transcription..."); trans_start = time.time(); result = model.transcribe(audio_path, word_timestamps=True, fp16=False); trans_end = time.time(); print(f"  DEBUG: Transcription finished in {trans_end - trans_start:.2f}s.")
        if not isinstance(result, dict) or "segments" not in result: print(f"❌ FAIL Step 2: Whisper result format unexpected."); return False
        print(f"  Saving results..."); lines_written = 0
        with open(output_transcription_path, "w", encoding="utf-8") as f:
            if not result["segments"]: print("  Warning: No segments found.")
            else:
                for segment in result["segments"]:
                    start = segment.get('start', 0.0); end = segment.get('end', 0.0); text = segment.get('text', '').strip()
                    if text: f.write(f"[{start:.2f} --> {end:.2f}] {text}\n"); lines_written += 1
        if not os.path.exists(output_transcription_path) or lines_written == 0: print(f"  Warn: Transcription file empty/not created?")
        else: print(f"Transcription saved ({lines_written} lines).")
        return True
    except Exception as e: print(f"❌ FAIL Step 2 (Transcription): {e}"); traceback.print_exc(); return False

# target_lang will now be passed from app.py
def translate_text(transcription_path, output_translation_path, target_lang, source_lang):
    print(f"\n--- Entering Step 3: Translate Text from '{source_lang}' to '{target_lang}' ---")
    if GoogleTranslator is None: print(f"❌ FAIL Step 3: Deep Translator missing."); return False
    if not os.path.exists(transcription_path): print(f"❌ FAIL Step 3: Transcription file missing: {transcription_path}"); return False
    
    # Handle auto detection logic for whisper transcription. If source_lang is 'auto', it means Whisper's output language is unknown.
    # The GoogleTranslator library automatically handles the 'auto' source language setting.
    
    try:
        if os.path.getsize(transcription_path) == 0: print(f"  Warn: Input empty. Creating empty file."); open(output_translation_path, 'w', encoding='utf-8').close(); return True
    except Exception as e: print(f"❌ FAIL Step 3 (File size check): {e}"); traceback.print_exc(); return False
    
    try:
        if os.path.exists(output_translation_path): print(f"'{output_translation_path}' exists. Skipping."); return True
        
        # KEY FIX: Pass source_lang code to the translator
        translator = GoogleTranslator(source=source_lang, target=target_lang); 
        translated_lines = []
        with open(transcription_path, "r", encoding="utf-8") as f: lines = f.readlines()
        total_lines = len(lines); print(f"  Translating {total_lines} lines...")
        
        for i, line in enumerate(lines):
            line = line.strip(); parts = line.split(']', 1);
            if not line or len(parts) != 2: continue
            timestamp, original_text = parts[0] + ']', parts[1].strip()
            if not original_text: translated_lines.append(f"{timestamp} "); continue
            
            try:
                time.sleep(0.12); 
                translated_text = translator.translate(original_text)
                if translated_text is None: translated_text = original_text
            except Exception as trans_error: 
                print(f"  Warn: Translation err line {i+1} (Source: {source_lang}, Text: '{original_text[:30]}...'): {trans_error}"); 
                translated_text = original_text
            
            translated_lines.append(f"{timestamp} {translated_text}")
            if (i + 1) % 10 == 0 or (i+1) == total_lines: print(f"    Translated {i+1}/{total_lines} lines...")
            
        with open(output_translation_path, "w", encoding="utf-8") as f:
            for line in translated_lines: f.write(line + "\n")
        print(f"Translation saved: '{output_translation_path}'"); return True
    except Exception as e: 
        print(f"❌ FAIL Step 3 (Translation): {e}"); traceback.print_exc(); return False

# target_lang will now be passed from app.py
def generate_tts_audio(translation_path, original_audio_path, output_tts_path, target_lang):
    print(f"\n--- Entering Step 4: Generate TTS Audio ('{target_lang}') ---")
    if gTTS is None or AudioSegment is None: print(f"❌ FAIL Step 4: gTTS/Pydub missing."); return False
    if not os.path.exists(translation_path): print(f"❌ FAIL Step 4: Translation file missing: {translation_path}"); return False
    if not os.path.exists(original_audio_path): print(f"❌ FAIL Step 4: Original audio missing: {original_audio_path}"); return False
    
    # 1. Check if the file already exists (NEW LOGIC)
    if os.path.exists(output_tts_path): 
        print(f"'{output_tts_path}' exists. Skipping.")
        # If the file exists, assume success and proceed immediately
        return True 

    try:
        if os.path.getsize(translation_path) == 0:
            print(f"  Warn: Input empty. Creating empty TTS file.")
            audio_info = ffmpeg.probe(original_audio_path); duration_ms = int(float(audio_info['format']['duration']) * 1000)
            AudioSegment.silent(duration=duration_ms).export(output_tts_path, format="mp3")
            return True
    except Exception as e: print(f"❌ FAIL Step 4 (File size check): {e}"); traceback.print_exc(); return False

    total_lines = 0; temp_files = []; success_flag = False
    try:
        audio_info = ffmpeg.probe(original_audio_path); duration_ms = int(float(audio_info['format']['duration']) * 1000)
        final_audio = AudioSegment.silent(duration=duration_ms)
        with open(translation_path, "r", encoding="utf-8") as f: lines = f.readlines()
        total_lines = len(lines); print(f"  Generating audio for {total_lines} lines...")
        for i, line in enumerate(lines):
            match = re.match(r'\[(\d+\.\d+) --> (\d+\.\d+)] (.*)', line)
            if not match: continue
            start_ms = time_str_to_ms(match.group(1)); tts_text = match.group(3).strip()
            
            if not tts_text: continue # FIX: Skip empty text
            
            temp_audio_file = f"temp_segment_{i}.mp3"
            try:
                tts = gTTS(text=tts_text, lang=target_lang); temp_files.append(temp_audio_file)
                tts.save(temp_audio_file);
                if not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0: print(f"  Warn: gTTS save failed line {i+1}."); continue
                segment_audio = AudioSegment.from_mp3(temp_audio_file)
                if start_ms < 0: start_ms = 0
                final_audio = final_audio.overlay(segment_audio, position=start_ms)
            except Exception as gtts_err:
                print(f"  Warn: gTTS error line {i+1} (Lang: {target_lang}, Text: '{tts_text}'): {gtts_err}");
                if temp_audio_file not in temp_files: temp_files.append(temp_audio_file)
            if (i + 1) % 10 == 0 or (i+1) == total_lines: print(f"    Generated audio for {i+1}/{total_lines} lines...")
        print(f"  Exporting final TTS: {output_tts_path}..."); final_audio.export(output_tts_path, format="mp3")
        print(f"TTS audio saved.")
        success_flag = True
    except Exception as e: 
        print(f"❌ FAIL Step 4 (TTS Generation): {e}"); traceback.print_exc(); success_flag = False
    finally:
        _cleanup_temp_files(temp_files, total_lines)
        return success_flag

def _cleanup_temp_files(temp_files_list, total_lines=None):
    print("  DEBUG: Cleaning up temp TTS files...")
    files_to_check = temp_files_list
    if not files_to_check and total_lines is not None and total_lines > 0: files_to_check = [f"temp_segment_{i}.mp3" for i in range(total_lines)]
    if files_to_check:
        for file_to_remove in files_to_check: 
            if os.path.exists(file_to_remove): 
                try: os.remove(file_to_remove)
                except Exception as e: print(f" Could not remove {file_to_remove}: {e}")

# --- SKIPPED Spleeter/Mixing original functions for speed ---
def separate_music(audio_path, spleeter_base_output_folder):
    print(f"\n--- Entering Step 5 (SKIPPED): Separate Music ---"); return None

# --- NEW FUNCTION 6: Audio Mixing (SIMPLE REPLACEMENT) ---
# NOTE: This ensures the output audio is ONLY the voice track, no dimming or background audio.
def mix_audio_with_dimming(voice_path, original_audio_path, output_mixed_path, mix_volume_db):
    print(f"\n--- Entering Step 6 (NEW - FINAL): Replacing Original Audio with Voice Only ---")
    if AudioSegment is None: print(f"❌ FAIL Step 6: Pydub missing."); return False
    if not os.path.exists(voice_path): print(f"❌ FAIL Step 6: Voice file missing: {voice_path}"); return False
    
    try:
        # Check if the mixed file exists, and if so, remove it to ensure fresh copy
        if os.path.exists(output_mixed_path):
            try:
                os.remove(output_mixed_path)
                print(f"  Removed old mixed audio file: {output_mixed_path}")
            except Exception as e:
                print(f"  Warning: Could not remove existing mixed audio file. Error: {e}")
                return False 
        
        # 1. Load the new TTS voice
        print(f"  Loading voice: {voice_path}")
        voice = AudioSegment.from_mp3(voice_path)
        
        # 2. Match length to original video length (optional, but safer)
        if os.path.exists(original_audio_path):
             original_len = len(AudioSegment.from_mp3(original_audio_path))
             if len(voice) < original_len:
                 voice = voice.append(AudioSegment.silent(duration=original_len - len(voice), frame_rate=voice.frame_rate), crossfade=0)
             elif len(voice) > original_len:
                 voice = voice[:original_len]

        # 3. Export the voice directly as the mixed output (NO DIMMING, NO ORIGINAL BG)
        print(f"  Exporting final TTS voice directly to mixed output: {output_mixed_path}...")
        voice.export(output_mixed_path, format="mp3")
        print(f"Mixed audio saved: '{output_mixed_path}'"); return True
        
    except Exception as e: 
        print(f"❌ FAIL Step 6 (New Audio Mixing): {e}"); traceback.print_exc(); return False


# --- FUNCTION 7: Merge Video (Double Audio Fix - FINAL MAPPING) ---
def merge_video(video_path, audio_path, output_video_path):
    print(f"\n--- Entering Step 7: Merge Video (ULTIMATE FIX: Two-Step Hard Mute) ---")
    if not os.path.exists(video_path): print(f"❌ FAIL Step 7: Input video missing: {video_path}"); return False
    if not os.path.exists(audio_path): print(f"❌ FAIL Step 7: Input audio missing: {audio_path}"); return False
    
    # Define a path for the temporary video clip without audio
    temp_video_no_audio = os.path.splitext(video_path)[0] + "_silent_temp.mp4"
    
    # We remove the existing final output file before running, just in case
    if os.path.exists(output_video_path):
        try: os.remove(output_video_path); print(f"  Removed old '{output_video_path}'.")
        except: print(f"  Warn: Could not remove old file.")
        
    process = None
    try:
        # Step 1: Create a temporary video with NO AUDIO from the original file
        print(f"  1. Creating temporary video without original audio (Using an=None)...")
        # Use an=None to explicitly strip ALL audio tracks
        (
            ffmpeg
            .input(video_path)
            .output(temp_video_no_audio, vcodec='copy', an=None) 
            .overwrite_output()
            .run(quiet=True, cmd=['ffmpeg', '-loglevel', 'warning'])
        )
        if not os.path.exists(temp_video_no_audio):
             print(f"❌ FAIL Step 7: Temp silent video was not created.")
             return False

        # Step 2: Merge the new mixed audio with the silent temporary video
        print(f"  2. Merging new mixed audio with the silent video...")
        
        input_video_silent = ffmpeg.input(temp_video_no_audio)
        input_audio_new = ffmpeg.input(audio_path)
        
        # We now merge, ensuring we only select streams from the silent input and the new audio input
        process = (
            ffmpeg
            .output(
                input_video_silent.video,   # Use ONLY the video stream from the silent temp file
                input_audio_new.audio,      # Use ONLY the audio stream from the new mixed file
                output_video_path,
                vcodec='copy',
                acodec='aac',
                strict='experimental'
            )
            .overwrite_output()
            .run_async(cmd=['ffmpeg', '-loglevel', 'warning'], pipe_stderr=True)
        )
        
        out, err = process.communicate()
        
        if process.returncode != 0:
            print(f"❌ FAIL Step 7 (ffmpeg error code {process.returncode}):")
            stderr = ""
            if err: 
                try: stderr = err.decode('utf-8', errors='ignore') 
                except: pass
            print(stderr); return False

        if not os.path.exists(output_video_path): print(f"❌ FAIL Step 7: Output video not created."); return False
        print(f"Final video saved: '{output_video_path}'"); return True
    
    except ffmpeg.Error as e:
        print(f"❌ FAIL Step 7 (ffmpeg specific error):"); stderr = ""
        if e.stderr: 
            try: stderr = e.stderr.decode('utf-8', errors='ignore') 
            except: pass
        print(stderr); return False
    except Exception as e: print(f"❌ FAIL Step 7 (general error): {e}"); traceback.print_exc(); return False
    finally:
        # Cleanup the temporary file
        if os.path.exists(temp_video_no_audio):
            try:
                os.remove(temp_video_no_audio)
                print(f"  Debug: Cleaned up temporary file: {temp_video_no_audio}")
            except:
                pass # Ignore cleanup errors


# --- FUNCTION 8: Generate Subtitles ---
def generate_subtitles(translation_path, output_subtitle_path):
    print(f"\n--- Entering Step 8: Generate Subtitles ---")
    if not os.path.exists(translation_path): print(f"❌ FAIL Step 8: Translation file missing: {translation_path}"); return False
    try:
        if os.path.getsize(translation_path) == 0: print(f"  Warn: Input empty. Creating empty file.");
        with open(output_subtitle_path, 'w', encoding='utf-8') as f: pass; return True
    except Exception as e: print(f"❌ FAIL Step 8 (File size check): {e}"); traceback.print_exc(); return False
    try:
        lines_written = 0
        with open(translation_path, "r", encoding="utf-8") as f_in, open(output_subtitle_path, "w", encoding="utf-8") as f_out:
            idx = 1
            for line in f_in:
                match = re.match(r'\[(\d+\.\d+) --> (\d+\.\d+)] (.*)', line)
                if not match: continue
                try: start_f = float(match.group(1)); end_f = float(match.group(2))
                except ValueError: print(f"  Warn: Could not parse time: {line.strip()}"); continue
                start_s = format_time_srt(start_f); end_s = format_time_srt(end_f)
                txt = match.group(3).strip()
                if not txt: continue
                f_out.write(f"{idx}\n{start_s} --> {end_s}\n{txt}\n\n"); idx += 1; lines_written += 1
        return True
    except Exception as e: print(f"❌ FAIL Step 8 (Generating Subtitles): {e}"); traceback.print_exc(); return False


# --- Main Execution Block (MODIFIED FOR SPEED & NEW MIXING) ---
def run_full_pipeline(input_vid, out_aud, out_trans, out_transl, out_tts, out_music, out_mixed, out_final, out_sub, model_name, target_lang, mix_volume_db, source_lang):
    success = False
    # NOTE: mix_volume_db is now ignored for a clean voice replacement
    print(f"\n>>> Running Pipeline (Optimized for Speed & Clean Voice Replacement) <<<") 
    
    out_trans_cleaned = out_trans.replace(".txt", "_cleaned.txt") 
    
    if extract_audio(input_vid, out_aud):
        if transcribe_audio(out_aud, out_trans, model_name=model_name):
            
            # Step 2.5: Clean Timing (Essential for avoiding double voice/overlap due to timing)
            if clean_transcription_timing(out_trans, out_trans_cleaned, min_gap_s=0.1): 
                
                # Step 3: Translate Text
                if translate_text(out_trans_cleaned, out_transl, target_lang=target_lang, source_lang=source_lang):
                    
                    # Step 4: Generate TTS Audio
                    if generate_tts_audio(out_transl, out_aud, out_tts, target_lang=target_lang):
                        
                        # Step 5/6: NEW MIXING LOGIC (Simple Replacement)
                        # We are now using the simplest, clean replacement to eliminate double voice.
                        if mix_audio_with_dimming(out_tts, out_aud, out_mixed, 0): # Pass 0 dB but logic ignores it
                        
                            # Step 7: Merge Video (Uses Hard Mute via two-step process)
                            if merge_video(input_vid, out_mixed, out_final):
                                
                                # Step 8: Generate Subtitles
                                if generate_subtitles(out_transl, out_sub): success = True
                                else: print("\n❌ FAIL: Subtitle Generation.")
                            else: print("\n❌ FAIL: Video Merging.")
                        else: print("\n❌ FAIL: Clean Audio Replacement.")
                        # --- END NEW MIXING ---

                    else: print("\n❌ FAIL: TTS Generation.")
                else: print("\n❌ FAIL: Translation.")
            else: print("\n❌ FAIL: Timing Cleanup.")
        else: print("\n❌ FAIL: Transcription.")
    else: print("\n❌ FAIL: Audio Extraction.")
    return success

if __name__ == '__main__':
    print("--- Starting AI Dubbing Tool Logic (Direct Run) ---")
    multiprocessing.freeze_support(); print("DEBUG: freeze_support() called.")

    INPUT_VID = "test_video.mp4";
    if not os.path.exists(INPUT_VID): print(f"FATAL: Input video '{INPUT_VID}' not found."); exit()
    OUT_AUD = "extracted_audio.mp3"; OUT_TRANS = "transcription_with_timestamps.txt"
    OUT_TRANSL = "translated_hindi_timestamps.txt"; OUT_TTS = "final_hindi_audio.mp3"
    os.makedirs(DEFAULT_SPLEETER_OUTPUT_FOLDER, exist_ok=True)
    OUT_MUSIC = os.path.join(DEFAULT_SPLEETER_OUTPUT_FOLDER, os.path.splitext(os.path.basename(OUT_AUD))[0], "accompaniment.wav")
    OUT_MIXED = "final_mixed_hindi_audio.mp3"; OUT_FINAL = DEFAULT_FINAL_VIDEO_FILE; OUT_SUB = DEFAULT_SUBTITLE_FILE

    start_time = time.time()
    # Dummy value for mix_volume_db in direct run
    success = run_full_pipeline(INPUT_VID, OUT_AUD, OUT_TRANS, OUT_TRANSL, OUT_TTS, OUT_MUSIC, OUT_MIXED, OUT_FINAL, OUT_SUB, model_name="tiny", target_lang='hi', mix_volume_db=-15, source_lang='en')
    end_time = time.time()
    print(f"\n--- Pipeline finished in {end_time - start_time:.2f} seconds ---")
    if success: print("✅✅✅ PIPELINE COMPLETED SUCCESSFULLY ✅✅✅")
    else: print("❌❌❌ PIPELINE FAILED ❌❌❌")

