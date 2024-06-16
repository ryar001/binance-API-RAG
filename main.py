import getpass
import os
import dotenv

dotenv.load_dotenv(dotenv_path=".env")  # Specify the path to your .env_ai file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", default=getpass.getpass())

breakpoint()
