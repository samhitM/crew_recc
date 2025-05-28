import requests
import time
import subprocess
import os
from dateutil import parser
from datetime import timezone
from datetime import datetime, timedelta


# ----------------------
# CONFIGURATION SECTION
# ----------------------

ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJWZXJzaW9uIjoiMi4wLjAuMCIsIktleVZlcnNpb24iOiI3NTExMjE1MGMzNDg0ZjI1ODdhNGFiMWE2OTMyMjE1OCIsIkFjY291bnRJZCI6ImQ5NDNlNTU0LWRmOWUtNDM1MS1hZjg2LTAxZTY5MDVlZTNjMSIsIkFjY291bnRUeXBlIjoiQXJtIiwiUGVybWlzc2lvbiI6IkNvbnRyaWJ1dG9yIiwiRXh0ZXJuYWxVc2VySWQiOiIzNUEwNUQ0NzA4MDA0RDNBOTdDNDU3QjI2NDYxMTBFRiIsIlVzZXJUeXBlIjoiTWljcm9zb2Z0Q29ycEFhZCIsIklzc3VlckxvY2F0aW9uIjoiYXVzdHJhbGlhZWFzdCIsIm5iZiI6MTc0ODEwMTIyMywiZXhwIjoxNzQ4MTA1MTIzLCJpc3MiOiJodHRwczovL2FwaS52aWRlb2luZGV4ZXIuYWkvIiwiYXVkIjoiaHR0cHM6Ly9hcGkudmlkZW9pbmRleGVyLmFpLyJ9.UP9xNlBqNB4EnR1udfwgIaUxH0qOv0PTciQXj2NXSnktB0XSVod31aEFexN0_N17xU8BF75mnkrFl-w5usS4e_Fry9LWGQpnZ4YlHzrqxwb1xhusu1-p3yighGWvVAm9SBGlIFb0n7Zq2Cgf23FtYfM9ASBU4iB8g1UL7fCpRJhbQViu8CMrT3kKLgh780HgBaLkAH--jM3FdsBePstNx0lj9en7sM6YTrlYvFqHMOmuei1HstnUmEbCT0qrB4shK01Q3pnXgYQWBx0J3hZUZRTBvMXxGPo7pgaJo3WFNl-C5ibgQxmE-rYySE7mWj-FgtDvYQ1ueXjLDdwElGBf-Q"
LOCATION = "australiaeast"
ACCOUNT_ID = "d943e554-df9e-4351-af86-01e6905ee3c1"
GIF_PATH = "Toon-Porn-Gif-3.gif"
MP4_PATH = "converted_video.mp4"
VIDEO_NAME = "moderation_test_video"

# ----------------------
# 1. CONVERT GIF TO MP4 USING FFMPEG
# ----------------------
def convert_gif_to_mp4(gif_path, mp4_path):
    print("[1] Converting GIF to MP4 using ffmpeg...")
    command = [
        "ffmpeg", "-y", "-i", gif_path,
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        mp4_path
    ]
    subprocess.run(command, check=True)
    print("Conversion complete.")

# ----------------------
# 2. UPLOAD VIDEO
# ----------------------
def upload_video(token, mp4_path, video_name):
    print("[2] Uploading video...")
    with open(mp4_path, 'rb') as f:
        files = {'file': f}
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
# 3. WAIT FOR PROCESSING
# ----------------------
def wait_for_processing(token, video_id):
    print("[3] Waiting for processing to complete...")
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
# 4. EXTRACT MODERATION INSIGHTS
# ----------------------
def extract_moderation_insights(data):
    print("[4] Extracting moderation insights...")

    try:
        # Access the correct nested path
        videos = data.get("videos", [])
        if not videos:
            print("No videos found in response.")
            return None

        video_insights = videos[0].get("insights", {})
        if not video_insights:
            print("No insights found inside video data.")
            return None

        # Print the visual and textual moderation sections
        visual = video_insights.get("visualContentModeration", [])
        textual = video_insights.get("textualContentModeration", {})

        print("Visual Content Moderation:")
        for item in visual:
            print(f"  - Adult Score: {item.get('adultScore', 0)}")
            print(f"  - Racy Score: {item.get('racyScore', 0)}")

        print("Textual Content Moderation:")
        print(f"  - Banned Words Count: {textual.get('bannedWordsCount', 0)}")

        return video_insights

    except Exception as e:
        print(f"❌ Failed to extract insights: {e}")
        return None


def flag_content(insights, thresholds=None):
    if insights is None:
        print("⚠️ No insights provided — flagging content by default.")
        return True

    if thresholds is None:
        thresholds = {
            "adult_score": 0.8,
            "racy_score": 0.8,
            "banned_words_count": 2,
        }

    flagged = False
    
    print(insights.get("visualContentModeration"))
    
    # Check visual content
    for item in insights.get("visualContentModeration", []):
        print(item.get("adultScore"))
        
        if item.get("adultScore", 0) >= thresholds["adult_score"]:
            print("⚠️ Flagged: Adult content detected.")
            flagged = True
        if item.get("racyScore", 0) >= thresholds["racy_score"]:
            print("⚠️ Flagged: Racy content detected.")
            flagged = True

    # Check textual content
    banned_words_count = insights.get("textualContentModeration", {}).get("bannedWordsCount", 0)
    if banned_words_count >= thresholds["banned_words_count"]:
        print("⚠️ Flagged: Inappropriate language detected.")
        flagged = True

    if not flagged:
        print("✅ Content is clean and within moderation thresholds.")

    return flagged

    

def delete_old_videos(token):
    print("[CLEANUP] Checking for old videos to delete...")
    response = requests.get(
        f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos",
        params={"accessToken": token}
    )
    response.raise_for_status()
    videos = response.json().get("results", [])

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)  # ✅ offset-aware

    deleted_count = 0
    for video in videos:
        video_id = video.get("id")
        name = video.get("name")
        created = video.get("created")
        created_time = parser.isoparse(created)

        if created_time < cutoff:
            print(f"Deleting '{name}' (created: {created_time.isoformat()})...")
            del_response = requests.delete(
                f"https://api.videoindexer.ai/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}",
                params={"accessToken": token}
            )
            del_response.raise_for_status()
            deleted_count += 1
            print(f"Deleted video ID {video_id}")

    if deleted_count == 0:
        print("No old videos found.")
    else:
        print(f"[CLEANUP COMPLETE] Deleted {deleted_count} old video(s).")
        
def log_moderation_instances(insights):
    print("[MODERATION LOG] Visual content moderation instances:")
    
    if not insights:
        print("⚠️ No insights provided.")
        return
    
    visual = insights.get("visualContentModeration", [])
    if not visual:
        print("✅ No visual moderation content detected.")
        return

    for idx, item in enumerate(visual, start=1):
        adult_score = item.get("adultScore", 0)
        racy_score = item.get("racyScore", 0)
        instances = item.get("instances", [])

        print(f"\nInstance #{idx}:")
        print(f"  - Adult Score: {adult_score}")
        print(f"  - Racy Score: {racy_score}")
        print(f"  - Time Ranges:")
        
        for inst in instances:
            start = inst.get("start", "Unknown")
            end = inst.get("end", "Unknown")
            print(f"      - From {start} to {end}")


# ----------------------
# MAIN FLOW
# ----------------------
def main():
    delete_old_videos(ACCESS_TOKEN)  # <-- clean up first
    convert_gif_to_mp4(GIF_PATH, MP4_PATH)
    token = ACCESS_TOKEN
    video_id = upload_video(token, MP4_PATH, VIDEO_NAME)
    data = wait_for_processing(token, video_id)
    
    insights = extract_moderation_insights(data)
    log_moderation_instances(insights)
    
    # Custom thresholds can be passed as needed
    thresholds = {
        "adult_score": 0.8,
        "racy_score": 0.8,
        "banned_words_count": 2
    }

    should_flag = flag_content(insights, thresholds)

    if should_flag:
        print("🚫 This video should be flagged for moderation.")
    else:
        print("✅ This video passed moderation checks.")


if __name__ == "__main__":
    main()
