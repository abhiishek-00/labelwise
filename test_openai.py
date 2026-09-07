import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    # 1. Check if the environment variable is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not found.")
        print("Please check that you have created a `.env` file containing: OPENAI_API_KEY=your_key")
        return

    print("🔄 Connecting to OpenAI...")
    try:
        # 2. Initialize the client (automatically reads OPENAI_API_KEY from environment)
        client = OpenAI()

        # 3. Send a lightweight chat completion request
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Highly cost-effective model for testing
            messages=[{"role": "user", "content": "Respond with the word 'Success!'"}],
            max_tokens=5
        )
        
        # 4. Print the result
        print(f"✅ Connection successful!")
        print(f"🤖 OpenAI Response: {response.choices[0].message.content.strip()}")

    except Exception as e:
        print(f"❌ Connection failed.")
        print(f"💬 Error details: {e}")

if __name__ == "__main__":
    test_connection()
