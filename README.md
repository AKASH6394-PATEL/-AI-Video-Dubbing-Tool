# AI Video Dubbing Tool 🎬🔊

This is a Python application that uses AI to automatically dub videos into multiple languages (like Hindi, Tamil, Telugu, etc.) while retaining the original background music and generating new subtitles.

The app is built with a Streamlit interface, making it easy to use:
1.  Upload a video.
2.  Select a target language.
3.  Select a Whisper model (smaller = faster).
4.  Click "Start Dubbing".
5.  Download the final dubbed video and subtitle file.

## 🤖 Tech Stack
-   **UI:** Streamlit
-   **Audio Extraction/Merging:** ffmpeg-python
-   **Transcription:** openai-whisper
-   **Translation:** deep-translator (Google Translate)
-   **Text-to-Speech:** gTTS
-   **Music Separation:** spleeter (TensorFlow)
-   **Audio Mixing:** pydub

---

## 🚀 How to Run This Project Locally

This is a complex project with many dependencies. Follow these steps carefully.

### 1. Prerequisites (Zaroori Cheezein)

-   **Python 3.9:** This project is tested and stable on **Python 3.9**. (Newer versions like 3.10+ may have issues with TensorFlow 2.9).
-   **ffmpeg.exe:** The `ffmpeg` program must be installed on your system and added to your PATH.
    -   Download link: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) (download the `ffmpeg-release-essentials.zip` file).
    -   Extract it and add the `bin` folder's location (e.g., `C:\ffmpeg\bin`) to your System Environment `Path` variable.
-   **C++ Build Tools:** `spleeter` (via `tensorflow`) needs C++ tools to build.
    -   Download link: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    -   Run the installer and select the **"Desktop development with C++"** workload.

### 2. Setup Virtual Environment

```bash
# Clone this repository (ya download zip)
git clone [https://github.com/AKASH6394-PATEL/AI-Video-Dubbing-Tool.git](https://github.com/AKASH6394-PATEL/AI-Video-Dubbing-Tool.git)
cd AI-Video-Dubbing-Tool

# Create a virtual environment using Python 3.9
# (Make sure 'py -3.9' points to your Python 3.9 installation)
py -3.9 -m venv venv

# Activate the virtual environment
venv\Scripts\activate

Install Dependencies
First, we install the large, complex libraries (tensorflow and protobuf) manually, and then the rest from requirements.txt.

# (Venv is active)

# 1. Install the specific TensorFlow 2.9.3 wheel (for Python 3.9)
# (This is the file we downloaded manually in our testing)
pip install [https://files.pythonhosted.org/packages/a9/0c/93e7f4c933b5c9006e331f79f0101037305c1f03fce7e7807a8b348b6c8f/tensorflow-2.9.3-cp39-cp39-win_amd64.whl](https://files.pythonhosted.org/packages/a9/0c/93e7f4c933b5c9006e331f79f0101037305c1f03fce7e7807a8b348b6c8f/tensorflow-2.9.3-cp39-cp39-win_amd64.whl)

# 2. Install the compatible protobuf version (for TensorFlow 2.9)
pip install "protobuf<3.20,>=3.9.2"

# 3. Install the compatible streamlit version (for old protobuf)
pip install "streamlit==1.22.0"

# 4. Now, install everything else from the requirements file
pip install -r requirements.txt

The app should now open in your browser at http://localhost:8501.


3.  Page par upar-right mein **"Commit changes"** waala hara button daba dijiye.

---

Bas! Ab aapka yeh project bhi GitHub par hai aur aap iska link (`https://github.com/AKASH6394-PATEL/AI-Video-Dubbing-Tool`) apne resume mein daal sakte hain.

Ab bataiye, naye project **(Task 3: Music Generation)** par chala jaaye?
