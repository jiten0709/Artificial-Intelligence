"""
### Pre-requisites for using Firestore with LangChain (Service Account Method):
1.  **Create a Firebase Account & Project:**
    *   Sign up at [firebase.google.com](https://firebase.google.com).
    *   Create a new project from the Firebase console.
    *   In the project, go to "Build" -> "Firestore Database" and create a database.

2.  **Enable Firestore API:**
    *   Ensure the Firestore API is enabled for your project in the Google Cloud Console.
    *   You can use this link (replace `YOUR_PROJECT_ID`): `https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=YOUR_PROJECT_ID`

3.  **Create a Service Account & JSON Key:**
    *   In the Google Cloud Console, go to "IAM & Admin" -> "Service Accounts".
    *   Select your project.
    *   Click **"+ CREATE SERVICE ACCOUNT"**.
    *   Give it a name (e.g., `firestore-chat-app`) and grant it the **`Cloud Datastore User`** role.
    *   Click "Done".
    *   Find the new account, click the "Actions" (⋮) menu, select "Manage keys", then "ADD KEY" -> "Create new key".
    *   Choose **JSON** and click "CREATE". A JSON key file will be downloaded.

4.  **Position the JSON Key File:**
    *   Place the downloaded JSON file into the path referenced by the script, for example: `Tool-Framework-Library/Langchain/assets/your-key-file.json`.
    *   Ensure the filename in the script matches the file you downloaded.

5.  **Install Python Dependencies:**
    *   `pip install langchain-google-firestore langchain-google-genai google-cloud-firestore google-auth python-dotenv`

"""

from dotenv import load_dotenv
load_dotenv()

from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2 import service_account

PROJECT_ID = "langchain-2fa27"
SESSION_ID = "langchain_session"
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
try:
    credentials = service_account.Credentials.from_service_account_file(
        'Tool-Framework-Library/Langchain/assets/langchain-2fa27-firebase-adminsdk-fbsvc-83be502fcb.json'
    )
    client = firestore.Client(project=PROJECT_ID, credentials=credentials)
    print("Successfully connected to Firestore using google-cloud-firestore library!")
except Exception as e:
    print(f"Failed to connect to Firestore: {e}")

# Initialize Firestore Chat Message History
try: 
    print("Initializing Firestore Chat Message History...")
    chat_history = FirestoreChatMessageHistory(
        session_id=SESSION_ID,
        collection=COLLECTION_NAME,
        client=client,
    )
    print("Chat History Initialized.")
    print("Current Chat History:", chat_history.messages)
except Exception as e:
    print(f"Failed to initialize Firestore Chat Message History: {e}")  
    

# Initialize Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
