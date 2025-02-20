import requests

def post_audio():
    # URL of your Flask endpoint
    url = "http://localhost:3001/api/transcribe/get"

    # Path to the audio file on your local system
    audio_file_path = "/home/ian/PycharmProjects/icad_transcript_api/test_audio/300-1739377770_155415000.0-call_5245.wav"

    # Form fields that you want to match your Flask endpoint
    # The keys here (audioName, audioType, dateTime, etc.)
    # should match the form fields your endpoint expects.
    data = {
        "key": "40812296-a338-4032-a5e1-4b65765fe8fb",
        "transcribe_config_id": 1,
        "sources": "[]"
    }

    # The 'files' dict is used to send multipart/form-data for the file upload.
    # The key should match the field name in request.files.get('audioFile')
    files = {
        "audio": open(audio_file_path, "rb")
    }

    # Send the POST request
    response = requests.post(url, files=files, data=data)

    # Print out the response for debugging
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Response Text:", response.text)

if __name__ == "__main__":
    post_audio()
