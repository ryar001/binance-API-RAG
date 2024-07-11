from openai import OpenAI
import getpass
import os
import dotenv


dotenv.load_dotenv(dotenv_path=".env")  # Specify the path to your .env_ai file
breakpoint()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the w·orld series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
)
breakpoint()
