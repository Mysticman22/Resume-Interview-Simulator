
import google.generativeai as genai

genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_prompt1, pdf_content[0], input_text])
    print(response)  # Check the raw response

