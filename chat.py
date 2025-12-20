from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    model = genai.GenerativeModel("gemini-2.0-pro")
    resp = model.generate_content("Say hello in one short sentence.")
    print(">>> OK — model responded:")
    print(resp.text)
except Exception as e:
    print(">>> ERROR calling model:")
    print(e)
