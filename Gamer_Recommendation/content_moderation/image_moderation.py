import requests
import base64
import json

# Set up Azure Content Safety details
SUBSCRIPTION_KEY = "At2YHsBDRtx0tlmJbZjVxEuqO0WxCkw8UU1fu5n8baF9sziSpFTaJQQJ99BDACL93NaXJ3w3AAAHACOG6V3J"
ENDPOINT = "https://crewcontentmoderator.cognitiveservices.azure.com"

HEADERS = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY
}

CATEGORIES = ["Hate", "SelfHarm", "Sexual", "Violence"]

def is_safe(response_json):
    return all(c["severity"] <= 2 for c in response_json.get("categoriesAnalysis", []))

def analyze_text(text):
    url = f"{ENDPOINT}/contentsafety/text:analyze?api-version=2024-09-01"
    payload = {
        "text": text,
        "categories": CATEGORIES
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    data = response.json()
    print("Text Response:", json.dumps(data, indent=2))
    return is_safe(data)

def analyze_image(image_path):
    url = f"{ENDPOINT}/contentsafety/image:analyze?api-version=2024-09-01"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "image": {
            "content": encoded_string
        },
        "categories": CATEGORIES
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    data = response.json()
    print("Image Response:", json.dumps(data, indent=2))
    return is_safe(data)

# === TESTS ===

# Example Text
sample_texts = [
    "I disagree with your opinion, but I respect your right to say it.",
    "Those people are worthless and should be eliminated.",
    "Sometimes I think about giving up, but I dont act on it.",
    "I want to hurt myself and end everything.",
    "He threatened to beat him up if he didn't cooperate.",
    "Im going to kill everyone in that place with a weapon.",
    "That was a pretty explicit scene for a TV show."
    "He forced himself on her and did disgusting things."
]

for text in sample_texts:
    print(f"\nAnalyzing: {text}")
    print("Safe?" if analyze_text(text) else "⚠️ Flagged")

image_result = analyze_image("Datasets/images/img13.png")
print("Is Image Safe?", image_result)