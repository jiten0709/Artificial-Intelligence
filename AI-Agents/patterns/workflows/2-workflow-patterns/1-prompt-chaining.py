import os
from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
import openai
import logging
import json

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

# --------------------------------------------------------------------
# Set up logging configuration
# --------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------------

class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""
    
    description: str = Field(
        description="Raw description of the event"
    )
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1"
    )

class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(
        description="Name of the event"
    )
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(
        description="Expected duration in minutes"
    )
    participants: list[str] = Field(
        description="List of participants"
    )

class EventConfirmatin(BaseModel): 
    """Third LLM call: Generate confirmation message"""

    confirmatin_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )

# --------------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------------

def extract_event_info(user_input: str) -> EventExtraction: 
    """First LLM call to determine if input is a calendar event"""

    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event. Respond as a JSON object with keys: description, is_calendar_event, confidence_score.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    result = EventExtraction(**json.loads(completion.choices[0].message.content))
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )

    return result

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""

    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract the event name, date, duration, and participants from the description. Respond as a JSON object with keys: name, date, duration_minutes, participants.",
            },
            {"role": "user", "content": description},
        ],
        response_format={"type": "json_object"},
    )
    result = EventDetails(**json.loads(completion.choices[0].message.content))
    logger.info(f"Event details parsed: {result}")

    return result

def generate_confirmation(event_details: EventDetails) -> EventConfirmatin:
    """Third LLM call to generate confirmation message"""

    logger.info("Generating confirmation message")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Generate a natural confirmation message for the event. Sign off with your name; Jiten. Respond as a JSON object with keys: confirmation_message, calendar_link.",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format={"type": "json_object"},
    )
    result = EventConfirmatin(**json.loads(completion.choices[0].message.content))
    logger.info(f"Confirmation generated: {result}")

    return result

# --------------------------------------------------------------------
# Step 3: chain the functions together
# --------------------------------------------------------------------

def process_calendar_request(user_input: str) -> Optional[EventConfirmatin]:
    """Main function implementing the prompt chain with gate check"""

    logger.info("Processing calendar request")
    logger.debug(f"User input: {user_input}")

    # Step 1: First LLM call: Extract event info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event 
        or initial_extraction.confidence_score < 0.7
    ): 
        logger.warning(
            f"Gate check failed: is_calendar_event={initial_extraction.is_calendar_event}, confidence_score={initial_extraction.confidence_score:.2f}"
        )
        return None
    
    logger.info("Gate check passed, proceeding to parse event details")

    # Step 2: Second LLM call: Parse event details
    event_details = parse_event_details(initial_extraction.description)

    # Step 3: Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)
    logger.info("Calendar request processing completed successfully")

    return confirmation

# --------------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------------

user_input = "Let's schedule a 1h team meeting next Friday at 7pm with Nyasa and Kiara to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmatin_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")

# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

invalid_input = "Can you send an email to Ayaan and Bob to discuss the project roadmap?"

result = process_calendar_request(invalid_input)
if result:
    print(f"Confirmation: {result.confirmatin_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")

    