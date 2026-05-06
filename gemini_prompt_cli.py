import os
from dotenv import load_dotenv

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with: pip install google-genai"
    ) from exc


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment.")

    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        raise SystemExit("Prompt is empty.")

    # Initialize the new client
    client = genai.Client(api_key=api_key)
    
    # Generate content using the new SDK syntax
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    text = response.text if response and response.text else ""

    print("\n--- Gemini 1.5 Flash Response ---\n")
    print(text.strip())


if __name__ == "__main__":
    main()