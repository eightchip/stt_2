from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
import requests
import logging
import os
import time
from pydub import AudioSegment
import moviepy.editor as mp

app = FastAPI()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

endpoint = 'https://acp-api-async.amivoice.com/v1/recognitions'

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
    filepath = os.path.join("/tmp", filename)  # Use /tmp directory on Vercel
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
