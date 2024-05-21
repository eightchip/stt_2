import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog
import requests
import pyperclip
import socket
import os
import time
import json
import urllib
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
import uvicorn
from dotenv import load_dotenv
from pydub import AudioSegment
import moviepy.editor as mp

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

# FastAPI setup
app = FastAPI()
endpoint = 'https://acp-api-async.amivoice.com/v1/recognitions'

PID_FILE = 'backend_server.pid'

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Extract audio from MP4 to WAV
def extract_audio_from_mp4(mp4_path, wav_path):
    video = mp.VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(wav_path)

# Handle audio file based on extension
def handle_audio_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".mp3":
        wav_path = file_path.replace(".mp3", ".wav")
        convert_mp3_to_wav(file_path, wav_path)
    elif file_ext == ".mp4":
        wav_path = file_path.replace(".mp4", ".wav")
        extract_audio_from_mp4(file_path, wav_path)
    elif file_ext == ".wav":
        wav_path = file_path
    else:
        raise ValueError("Unsupported file format")
    return wav_path

# FastAPI endpoint for audio recognition
@app.post("/recognize/")
async def recognize_audio(
    background_tasks: BackgroundTasks,
    app_key: str = Form(...),
    file: UploadFile = File(...),
    filename: str = Form(...),
    grammarFileNames: str = Form(...),
    loggingOptOut: str = Form(...),
    contentId: str = Form(...),
    speakerDiarization: str = Form(...),
    diarizationMinSpeaker: int = Form(...),
    diarizationMaxSpeaker: int = Form(...),
    profileId: str = Form(None)
):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())

    logger.debug(f"Using API key: {app_key}")

    domain = {
        'grammarFileNames': grammarFileNames,
        'loggingOptOut': loggingOptOut,
        'contentId': contentId,
        'speakerDiarization': speakerDiarization,
        'diarizationMinSpeaker': str(diarizationMinSpeaker),
        'diarizationMaxSpeaker': str(diarizationMaxSpeaker)
    }
    
    if profileId:
        domain['profileId'] = profileId

    params = {
        'u': app_key,
        'd': ' '.join([f'{key}={urllib.parse.quote(value)}' for key, value in domain.items()]),
    }
    logger.info(params)

    try:
        with open(filepath, 'rb') as f:
            request_response = requests.post(
                url=endpoint,
                data={key: value for key, value in params.items()},
                files={'a': (filename, f.read(), 'application/octet-stream')}
            )
        
        if request_response.status_code != 200:
            logger.error(f'Failed to request - {request_response.content}')
            raise HTTPException(status_code=request_response.status_code, detail="Failed to create job")
        
        request = request_response.json()
        
        if 'sessionid' not in request:
            logger.error(f'Failed to create job - {request["message"]} ({request["code"]})')
            raise HTTPException(status_code=400, detail="Failed to create job")

        logger.info(request)
        background_tasks.add_task(check_status, request["sessionid"], app_key, endpoint, filepath)
        return {"message": "Job started", "sessionid": request["sessionid"]}
    except Exception as e:
        logger.error(f'An error occurred: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status/{sessionid}")
async def get_status(sessionid: str, app_key: str):
    try:
        logger.info(f"Checking status for session ID: {sessionid}")
        result_response = requests.get(
            url=f'{endpoint}/{sessionid}',
            headers={'Authorization': f'Bearer {app_key}'}
        )
        if result_response.status_code == 200:
            result = result_response.json()
            logger.info(f"Received result: {result}")
            return result
        else:
            logger.error(f'Failed. Response is {result_response.content}')
            raise HTTPException(status_code=result_response.status_code, detail="Failed to get status")
    except Exception as e:
        logger.error(f'An error occurred while checking status: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")

def check_status(sessionid: str, app_key: str, endpoint: str, filepath: str):
    while True:
        try:
            logger.info(f"Checking status for session ID: {sessionid}")
            result_response = requests.get(
                url=f'{endpoint}/{sessionid}',
                headers={'Authorization': f'Bearer {app_key}'}
            )
            if result_response.status_code == 200:
                result = result_response.json()
                logger.info(f"Received result: {result}")

                if 'status' in result and (result['status'] == 'completed' or result['status'] == 'error'):
                    logger.info(json.dumps(result, ensure_ascii=False, indent=4))
                    if result['status'] == 'completed':
                        if 'text' in result:
                            text_output_path = filepath.rsplit('.', 1)[0] + ".txt"
                            with open(text_output_path, 'w', encoding='utf-8') as f:
                                f.write(result['text'])
                            logger.info(f"Transcription saved to {text_output_path}")
                            notify_client(filepath, text_output_path)
                            os._exit(0)  # プログラムを終了
                        else:
                            logger.error("The 'text' key is missing in the API response.")
                    break
                else:
                    logger.info(f"Status: {result['status']}. Retrying in 20 seconds.")
                    time.sleep(20)
            else:
                logger.error(f'Failed. Response is {result_response.content}')
                break
        except Exception as e:
            logger.error(f'An error occurred while checking status: {e}')
            break

def notify_client(filepath: str, text_output_path: str):
    notification_file = filepath.rsplit('.', 1)[0] + "_completed.txt"
    with open(notification_file, 'w', encoding='utf-8') as f:
        f.write(f"Transcription completed and saved to {text_output_path}")

# GUI and backend process management
def is_backend_running():
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            return True
    except OSError:
        return False

def start_backend():
    if not is_backend_running():
        # Create a new process for the backend server
        process = subprocess.Popen([sys.executable, __file__, '--start-server'])
        # Write the PID to a file
        with open(PID_FILE, 'w') as f:
            f.write(str(process.pid))

def recognize_audio(api_key, file_path, logging_opt_out, min_speaker, max_speaker, grammar_file, profile_id):
    file_path = handle_audio_file(file_path)  # ここで適切な変換を行う
    url = "http://127.0.0.1:8000/recognize/"
    files = {'file': open(file_path, 'rb')}
    data = {
        'app_key': api_key,
        'filename': os.path.basename(file_path),
        'grammarFileNames': grammar_file,
        'loggingOptOut': logging_opt_out,
        'contentId': os.path.basename(file_path),
        'speakerDiarization': 'True',
        'diarizationMinSpeaker': str(min_speaker),
        'diarizationMaxSpeaker': str(max_speaker),
        'profileId': profile_id
    }
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        result = response.json()
        sessionid = result.get("sessionid")
        result_text.set("Processing")
        threading.Thread(target=check_completion, args=(api_key, sessionid, file_path)).start()
    else:
        result_text.set("Failed: " + str(response.status_code) + " " + response.text)

def check_completion(api_key, sessionid, file_path):
    url = f"http://127.0.0.1:8000/status/{sessionid}?app_key={api_key}"
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'completed':
                text_output_path = file_path.rsplit('.', 1)[0] + ".txt"
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('text', ''))
                result_text.set(f"Transcription completed and saved to {text_output_path}")
                pyperclip.copy(open(text_output_path).read())
                os._exit(0)  # プログラムを終了
                break
            elif result.get('status') == 'error':
                result_text.set("Error in processing audio")
                break
            else:
                time.sleep(10)
        else:
            result_text.set("Failed to get status")
            break

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        recognize_audio(api_key_entry.get(), file_path, logging_opt_out_var.get(), min_speaker_scale.get(), max_speaker_scale.get(), grammar_file_var.get(), profile_id_entry.get())

def create_gui():
    global api_key_entry, logging_opt_out_var, min_speaker_scale, max_speaker_scale, grammar_file_var, profile_id_entry, result_text, root
    root = tk.Tk()
    root.title("Audio Recognizer")
    root.geometry("700x600")  # 縦の幅を小さくしました

    frame = tk.Frame(root)
    frame.pack(pady=20, padx=20)

    tk.Label(frame, text="API Key:").grid(row=0, column=0, padx=10, pady=5)
    api_key_entry = tk.Entry(frame, width=50)
    api_key_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(frame, text="Profile ID:").grid(row=1, column=0, padx=10, pady=5)
    profile_id_entry = tk.Entry(frame, width=10)
    profile_id_entry.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(frame, text="使用エンジン(日・英・韓・中)").grid(row=2, column=0, padx=10, pady=5)
    grammar_file_var = tk.StringVar(value="-a-general")
    grammar_options = ["-a-general", "-a-general-en", "-a-general-ko", "-a-general-zh"]
    tk.OptionMenu(frame, grammar_file_var, *grammar_options).grid(row=2, column=1, padx=10, pady=5)

    logging_opt_out_var = tk.StringVar(value='true')
    tk.Label(frame, text="ログを保存しない:").grid(row=3, column=0, padx=10, pady=5)
    tk.Radiobutton(frame, text="保存しません", variable=logging_opt_out_var, value='true').grid(row=3, column=1, padx=10, pady=5)
    tk.Radiobutton(frame, text="保存します", variable=logging_opt_out_var, value='false').grid(row=3, column=2, padx=10, pady=5)

    tk.Label(frame, text="最小話者数:").grid(row=4, column=0, padx=10, pady=5)
    min_speaker_scale = tk.Scale(frame, from_=1, to=10, orient="horizontal")
    min_speaker_scale.set(2)
    min_speaker_scale.grid(row=4, column=1, padx=10, pady=5)

    tk.Label(frame, text="最大話者数:").grid(row=5, column=0, padx=10, pady=5)
    max_speaker_scale = tk.Scale(frame, from_=1, to=20, orient="horizontal")
    max_speaker_scale.set(4)
    max_speaker_scale.grid(row=5, column=1, padx=10, pady=5)

    tk.Button(frame, text="ファイル選択", command=open_file, height=3, width=20, bg="blue", fg="white").grid(row=6, column=0, columnspan=3, pady=20)

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, wraplength=500, justify="left")
    result_label.pack(pady=20)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

def on_close():
    if backend_thread.is_alive():
        backend_thread.join()  # Wait for the thread to finish
    # Read the PID from the file
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read())
        # Terminate the backend process
        os.kill(pid, 9)
    except Exception as e:
        logger.error(f"Failed to terminate backend process: {e}")
    root.destroy()  # Close the GUI

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-server', action='store_true', help='Start the FastAPI server')
    args = parser.parse_args()

    if args.start_server:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        if not is_backend_running():
            backend_thread = threading.Thread(target=start_backend, daemon=True)
            backend_thread.start()
            time.sleep(3)
        create_gui()
