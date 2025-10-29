import streamlit as st
import os
import time
import re
import traceback
import glob # For cleanup
import shutil # For rmtree

# --- Check logic file ---
logic_file_path = "ai_dubbing_tool_logic.py"
if not os.path.exists(logic_file_path):
    st.error(f"FATAL ERROR: Logic file '{logic_file_path}' not found! Make sure it's in the same folder as app.py.")
    st.stop()

# --- Import logic functions ---
try:
    from ai_dubbing_tool_logic import (
        run_full_pipeline,
        DEFAULT_SPLEETER_OUTPUT_FOLDER
    )
    print("DEBUG: Logic functions imported successfully.")
except Exception as e:
    st.error(f"FATAL ERROR importing from '{logic_file_path}': {e}")
    traceback.print_exc(); st.stop()
# ---------------------------------------------------

# --- Streamlit App Setup ---
st.set_page_config(page_title="AI Dubbing Tool", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Dubbing Tool")
st.write("Fast Dubbing, Timing Fixes, and Professional Audio Mixing.")

# --- Directory Setup ---
TEMP_DIR = "temp_streamlit_uploads"
OUTPUT_DIR = "output_videos"
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_SPLEETER_OUTPUT_FOLDER, exist_ok=True)
except OSError as e: st.error(f"Error creating directories: {e}"); st.stop()

# --- Language & Model Selection ---
LANGUAGES = { # Language Name -> Code for deep-translator and gTTS
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
}

st.header("⚙️ Processing Settings")
col_src, col_target, col_model = st.columns(3)

with col_src:
    source_lang_name = st.selectbox(
        "1. Select Video Source Language:",
        list(LANGUAGES.keys()), 
        index=0 # Default to English
    )
    source_lang_code = LANGUAGES[source_lang_name]

with col_target:
    target_lang_name = st.selectbox(
        "2. Select Target Dubbing Language:",
        list(LANGUAGES.keys()), 
        index=4 # Default to Hindi
    )
    target_lang_code = LANGUAGES[target_lang_name]

with col_model:
    whisper_model_name = st.selectbox(
        "3. Select Transcription Model (For Speed, use 'tiny'):",
        ('tiny', 'base', 'small', 'medium'), 
        index=0 # Default to 'tiny' for speed
    )

st.header("🔊 Audio Control Settings (Mixer)")
col_audio1, col_audio2 = st.columns(2)

with col_audio1:
    # Mixer Value: Decibels. Lower is quieter.
    audio_dim_db = st.slider(
        "4. Original Background Audio Volume (dB):",
        min_value=-30, # -30 dB is very quiet
        max_value=-5,  # -5 dB is moderately quiet
        value=-15,     # Default to -15 dB (Good background level)
        step=1,
        help="Sets how much the original audio (music/sfx/speech) should be quieted down behind the new dubbed voice. -30 is very quiet; -5 is loud."
    )

st.info(f"Source Language: **{source_lang_name}**. Target Language: **{target_lang_name}** ({target_lang_code}). Model: **{whisper_model_name}**. Mix Volume: **{audio_dim_db} dB**.")

# --- File Uploader ---
uploaded_file = st.file_uploader("5. Upload your Video File (.mp4):", type=["mp4"], key="file_uploader")

# --- Processing Logic ---
if uploaded_file is not None:
    try:
        # Create safe base name using regex to ensure no invalid characters in file paths
        safe_base_name = re.sub(r'[^\w\-]+', '_', os.path.splitext(uploaded_file.name)[0])
        temp_video_path = os.path.join(TEMP_DIR, f"{safe_base_name}_original.mp4")

        # Define all pipeline file paths
        extracted_audio_path = os.path.join(TEMP_DIR, f"{safe_base_name}_audio.mp3")
        transcription_path = os.path.join(TEMP_DIR, f"{safe_base_name}_transcription.txt")
        translation_path = os.path.join(TEMP_DIR, f"{safe_base_name}_translation_{target_lang_code}.txt")
        tts_audio_path = os.path.join(TEMP_DIR, f"{safe_base_name}_tts_{target_lang_code}.mp3")
        
        spleeter_subfolder_name = os.path.splitext(os.path.basename(extracted_audio_path))[0]
        music_path = os.path.join(DEFAULT_SPLEETER_OUTPUT_FOLDER, spleeter_subfolder_name, "accompaniment.wav")
        
        # NOTE: We use mixed_audio_path for the combined TTS + Dimmed Original Audio
        mixed_audio_path = os.path.join(TEMP_DIR, f"{safe_base_name}_mixed_audio_{target_lang_code}.mp3")
        
        final_video_path = os.path.join(OUTPUT_DIR, f"{safe_base_name}_{target_lang_name.upper()}_DUBBED.mp4")
        subtitle_path = os.path.join(OUTPUT_DIR, f"{safe_base_name}_{target_lang_name.upper()}_SUBTITLES.srt")
    except Exception as path_e: st.error(f"Error defining file paths: {path_e}"); st.stop()

    st.write(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")

    if st.button("🏁 Start Dubbing Process", key="start_button"):
        
        # --- Pre-processing/File Write ---
        try:
            # Remove previous runs of the final file
            if os.path.exists(final_video_path): os.remove(final_video_path)
            if os.path.exists(subtitle_path): os.remove(subtitle_path)
            
            # Write uploaded file to temporary disk
            with open(temp_video_path, "wb") as f: f.write(uploaded_file.getbuffer())
            st.success(f"Video saved temporarily.")
        except Exception as e: st.error(f"Error saving uploaded file: {e}"); st.stop()

        start_full_time = time.time(); pipeline_successful = False
        st.write("--- Starting Processing ---")

        # --- Run Pipeline ---
        with st.spinner(f"Running full dubbing pipeline for {target_lang_name}..."):
            pipeline_successful = run_full_pipeline(
                input_vid=temp_video_path,
                out_aud=extracted_audio_path,
                out_trans=transcription_path,
                out_transl=translation_path,
                out_tts=tts_audio_path,
                out_music=DEFAULT_SPLEETER_OUTPUT_FOLDER, # Unused path, kept for compatibility
                out_mixed=mixed_audio_path,
                out_final=final_video_path,
                out_sub=subtitle_path,
                model_name=whisper_model_name,
                target_lang=target_lang_code,
                mix_volume_db=audio_dim_db, # Pass the user's chosen volume setting
                source_lang=source_lang_code # Pass the new source language setting
            )

        st.write("--- Processing Finished ---")
        end_full_time = time.time()
        st.info(f"Total processing time: {end_full_time - start_full_time:.2f} seconds")

        if pipeline_successful:
            st.balloons()
            st.header("✅ Dubbing Completed Successfully!")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if os.path.exists(final_video_path):
                    try:
                        with open(final_video_path, "rb") as fp_video:
                            st.download_button(label=f"⬇️ Download {target_lang_name} Dubbed Video (.mp4)", data=fp_video,
                                               file_name=os.path.basename(final_video_path), mime="video/mp4")
                    except Exception as e: st.error(f"Error reading video file: {e}")
                else: st.error(f"Could not find final video.")
            
            with col_dl2:
                if os.path.exists(subtitle_path):
                    try:
                        with open(subtitle_path, "rb") as fp_srt:
                            st.download_button(label=f"⬇️ Download {target_lang_name} Subtitles (.srt)", data=fp_srt,
                                               file_name=os.path.basename(subtitle_path), mime="text/plain")
                    except Exception as e: st.error(f"Error reading subtitle file: {e}")
                else: st.warning(f"Could not find subtitle file.")
        else:
            st.error("Pipeline failed. Check terminal output for errors.")

        st.write("---")
        # --- Cleanup button functionality ---
        if st.button("Clean Temporary Files (Recommended)", key=f"clean_button_{int(time.time())}"):
            files_to_clean = [
                temp_video_path, extracted_audio_path, transcription_path,
                translation_path, tts_audio_path, mixed_audio_path
            ]
            
            # Add cleanup for transcription_with_timestamps_cleaned.txt (new file in Step 2.5)
            trans_cleaned_path = os.path.join(TEMP_DIR, f"{safe_base_name}_transcription_cleaned.txt")
            if os.path.exists(trans_cleaned_path):
                 files_to_clean.append(trans_cleaned_path)
            
            # Clean temporary segment files
            temp_tts_files = glob.glob(os.path.join(os.getcwd(), "temp_segment_*.mp3"))
            files_to_clean.extend(temp_tts_files)
            
            cleaned_count = 0; failed_files_list = []
            st.write("Cleaning temporary files...")
            for f_path in files_to_clean:
                if f_path and os.path.exists(f_path):
                    try: os.remove(f_path); cleaned_count += 1
                    except Exception as clean_err: failed_files_list.append(os.path.basename(f_path)); st.warning(f"Could not remove {os.path.basename(f_path)}: {clean_err}")
            
            # Attempt to remove the spleeter subfolder if it exists (though unused now)
            spleeter_sub_dir = os.path.join(DEFAULT_SPLEETER_OUTPUT_FOLDER, spleeter_subfolder_name)
            if os.path.isdir(spleeter_sub_dir):
                try: shutil.rmtree(spleeter_sub_dir)
                except: pass
                
            st.success(f"Removed {cleaned_count} temporary files. Please restart the app.")
            if failed_files_list:
                with st.expander("Failed to Remove"): st.write(", ".join(failed_files_list))
            st.warning("Final output files (MP4/SRT) and base folders were not removed.")

else:
    st.info("⬆️ Upload a video file (.mp4) to start the dubbing process.")

# --- Sidebar Info ---
st.sidebar.header("About This Tool")
st.sidebar.info("This tool performs high-speed video dubbing using open-source technologies (Whisper, gTTS, FFmpeg) and includes advanced timing and audio mixing controls.")
st.sidebar.header("⚠️ Notes")
st.sidebar.warning("1. Transcription and Translation require a stable internet connection.\n2. Use the 'tiny' model for maximum speed on CPU systems.")

# --- End of File ---