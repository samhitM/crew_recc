import requests
import time
import os
from moviepy.editor import VideoFileClip

# ----------------------
# CONFIGURATION SECTION
# ----------------------

SUBSCRIPTION_KEY = "YOUR_PRIMARY_API_KEY"  # Replace with your Azure Video Indexer API key
LOCATION = "trial"  # or actual Azure region like "southeastasia"
ACCOUNT_ID = "3b86d08f-aefe-4243-8555-cc6b3b37f271" 
GIF_PATH = "example.gif"  # Input GIF path
MP4_PATH = "converted_video.mp4"  # Output video path
VIDEO_NAME = "moderation_test_video"

# ----------------------
# 1. CONVERT GIF TO MP4
# ----------------------
def convert_gif_to_mp4(gif_path, mp4_path):
    print("[1] Converting GIF to MP4...")
    clip = VideoFileClip(gif_path)
    clip.write_videofile(mp4_path, codec="libx264")
    print("Conversion complete.")

# ----------------------
# 2. GET ACCESS TOKEN
# ----------------------
def get_access_token():
    print("[2] Fetching access token...")
    headers = {
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY
    }
    response = requests.get(
        f"https://api.videoindexer.ai/Auth/{LOCATION}/Accounts/{ACCOUNT_ID}/AccessToken",
        headers=headers
    )
    response.raise_for_status()
    token = response.text.strip('"')
    print("Access token received.")
    return token

# ----------------------
# 3. UPLOAD VIDEO
# ----------------------
def upload_video(token, mp4_path, video_name):
    print("[3] Uploading video...")
    files = {'file': open(mp4_path, 'rb')}
    params = {
        'name': video_name,
        'accessToken': token
    }
    response = requests.post(
        f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos",
        files=files,
        params=params
    )
    response.raise_for_status()
    video_id = response.json()["id"]
    print(f"Upload complete. Video ID: {video_id}")
    return video_id

# ----------------------
# 4. WAIT FOR PROCESSING
# ----------------------
def wait_for_processing(token, video_id):
    print("[4] Waiting for processing to complete...")
    while True:
        response = requests.get(
            f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}/Index",
            params={"accessToken": token}
        )
        response.raise_for_status()
        data = response.json()
        state = data.get("state", "Unknown")
        print("Current state:", state)
        if state == "Processed":
            print("Processing complete.")
            return data
        elif state == "Failed":
            raise RuntimeError("Processing failed.")
        time.sleep(10)

# ----------------------
# 5. EXTRACT MODERATION INSIGHTS
# ----------------------
def extract_moderation_insights(data):
    print("[4] Extracting moderation insights...")

    try:
        video_data = data.get("videos", [])[0]
        insights = video_data.get("insights", {})

        # Visual moderation
        visual = insights.get("visualContentModeration", [])
        textual = insights.get("textualContentModeration", {})

        if visual:
            print("Visual Moderation Results:")
            for item in visual:
                print(f"- Adult Score: {item.get('adultScore')}")
                print(f"  Racy Score: {item.get('racyScore')}")
                for instance in item.get("instances", []):
                    print(f"  Start: {instance.get('start')}, End: {instance.get('end')}")
        else:
            print("No visual content moderation issues detected.")

        if textual and textual.get("bannedWordsCount", 0) > 0:
            print("Textual Moderation Results:")
            print(f"- Banned Words Count: {textual.get('bannedWordsCount')}")
        else:
            print("No textual content moderation issues detected.")

    except Exception as e:
        print(f"Error extracting moderation insights: {e}")


# ----------------------
# MAIN FLOW
# ----------------------
def main():
    convert_gif_to_mp4(GIF_PATH, MP4_PATH)
    token = get_access_token()
    video_id = upload_video(token, MP4_PATH, VIDEO_NAME)
    insights_data = wait_for_processing(token, video_id)
    extract_moderation_insights(insights_data)

if __name__ == "__main__":
    main()
