import os
from dotenv import load_dotenv
import uuid
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, ValidationError
import openai
import logging
import json

# --------------------------------------------------------------------
# Configuration Management
# --------------------------------------------------------------------
# Configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MODEL = "gpt-4o-mini"
DATE_FORMAT = "%A, %B %d, %Y"

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

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

class EventConfirmation(BaseModel): 
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )

# --------------------------------------------------------------------
# Helper functions for robust date handling
# --------------------------------------------------------------------

def parse_relative_date(date_text: str, reference_date: datetime = None) -> datetime:
    """
    Parse relative date expressions like 'next Friday' into datetime objects.
    
    Args:
        date_text: Text describing the date
        reference_date: Reference date (defaults to today)
        
    Returns:
        datetime object
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    try:
        # Try parsing directly first
        return parser.parse(date_text, fuzzy=True, default=reference_date)
    except (ValueError, parser.ParserError):
        # Handle specific relative date patterns
        date_text = date_text.lower()
        
        # Handle "next X" patterns
        if "next" in date_text:
            days = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }
            
            for day, day_num in days.items():
                if day in date_text:
                    days_ahead = (day_num - reference_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7  # If today is the day, go to next week
                    return reference_date + timedelta(days=days_ahead)
            
            # Handle "next week"
            if "week" in date_text:
                return reference_date + timedelta(days=7)
            
            # Handle "next month"
            if "month" in date_text:
                return reference_date + relativedelta(months=1)
                
        # Return the reference date if we can't parse
        logger.warning(f"Could not parse date: {date_text}, using reference date")
        return reference_date

def format_as_iso8601(dt: datetime) -> str:
    """Format datetime as ISO 8601 string"""
    return dt.isoformat()

# --------------------------------------------------------------------
# Structured output formatting
# --------------------------------------------------------------------

def format_event_confirmation(confirmation: EventConfirmation) -> str:
    """Format event confirmation in a user-friendly way"""
    
    output = []
    output.append("📅 EVENT CONFIRMATION 📅")
    output.append("-" * 30)
    output.append(confirmation.confirmation_message)
    
    if confirmation.calendar_link:
        output.append("\n📎 Add to calendar:")
        output.append(confirmation.calendar_link)
    
    return "\n".join(output)

# --------------------------------------------------------------------
# Step 2: Define the functions with error handling
# --------------------------------------------------------------------

def make_llm_call(messages: list, response_model: BaseModel = None) -> Union[Dict[str, Any], BaseModel]:
    """
    Make an LLM API call with error handling
    
    Args:
        messages: List of message objects for the API call
        response_model: Optional Pydantic model to validate response
        
    Returns:
        Validated model instance or raw response
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Making LLM API call")
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        
        response_json = json.loads(completion.choices[0].message.content)
        
        if response_model:
            try:
                return response_model(**response_json)
            except ValidationError as e:
                logger.error(f"[{request_id}] Validation error: {str(e)}")
                raise ValueError(f"Response validation failed: {str(e)}")
        
        return response_json
    
    except openai.APIError as e:
        logger.error(f"[{request_id}] OpenAI API error: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] JSON parsing error: {str(e)}")
        raise ValueError(f"Failed to parse JSON response: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise

def extract_event_info(user_input: str) -> EventExtraction: 
    """First LLM call to determine if input is a calendar event"""

    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime(DATE_FORMAT)}."

    try:
        messages = [
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event. Respond as a JSON object with keys: description, is_calendar_event, confidence_score.",
            },
            {"role": "user", "content": user_input},
        ]
        
        result = make_llm_call(messages, EventExtraction)
        
        logger.info(
            f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to extract event info: {str(e)}")
        # Create a fallback response for graceful failure
        return EventExtraction(
            description=user_input,
            is_calendar_event=False,
            confidence_score=0.0
        )

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""

    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime(DATE_FORMAT)}."

    try:
        messages = [
            {
                "role": "system",
                "content": f"{date_context} Extract the event name, date, duration, and participants from the description. For dates, first understand relative references like 'next Friday'. Respond as a JSON object with keys: name, date, duration_minutes, participants.",
            },
            {"role": "user", "content": description},
        ]
        
        result = make_llm_call(messages, EventDetails)
        
        # Validate and potentially correct the date format
        try:
            parsed_date = parser.parse(result.date)
            result.date = format_as_iso8601(parsed_date)
        except (ValueError, parser.ParserError):
            # Try to handle relative dates if ISO parsing fails
            try:
                parsed_date = parse_relative_date(result.date, today)
                result.date = format_as_iso8601(parsed_date)
            except Exception as date_error:
                logger.warning(f"Could not parse date properly: {date_error}")
        
        logger.info(f"Event details parsed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Failed to parse event details: {str(e)}")
        raise

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate confirmation message"""

    logger.info("Generating confirmation message")

    today = datetime.now()
    date_context = f"Today is {today.strftime(DATE_FORMAT)}."

    try:
        # Parse the event date for better context
        try:
            event_date = parser.parse(event_details.date)
            formatted_date = event_date.strftime("%A, %B %d at %I:%M %p")
        except:
            formatted_date = event_details.date
        
        messages = [
            {
                "role": "system",
                "content": f"{date_context}. Generate a natural confirmation message for the event scheduled for {formatted_date}. Sign off with your name; Jiten. Respond as a JSON object with keys: confirmation_message, calendar_link.",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ]
        
        result = make_llm_call(messages, EventConfirmation)
        
        # Generate a calendar link if one wasn't provided
        if not result.calendar_link:
            try:
                event_date = parser.parse(event_details.date)
                end_date = event_date + timedelta(minutes=event_details.duration_minutes)
                
                # Create a Google Calendar link
                result.calendar_link = (
                    f"https://calendar.google.com/calendar/render?action=TEMPLATE"
                    f"&text={event_details.name.replace(' ', '+')}"
                    f"&dates={event_date.strftime('%Y%m%dT%H%M%S')}/{end_date.strftime('%Y%m%dT%H%M%S')}"
                    f"&details=Event+with+{'+'.join(event_details.participants)}"
                )
            except Exception as calendar_error:
                logger.warning(f"Could not generate calendar link: {calendar_error}")
        
        logger.info(f"Confirmation generated")
        return result
    
    except Exception as e:
        logger.error(f"Failed to generate confirmation: {str(e)}")
        raise

# --------------------------------------------------------------------
# Step 3: chain the functions together
# --------------------------------------------------------------------

def process_calendar_request(
    user_input: str, 
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing calendar request")
    logger.debug(f"[{request_id}] User input: {user_input}")

    try:
        # Step 1: First LLM call: Extract event info
        initial_extraction = extract_event_info(user_input)

        # Gate check: Verify if it's a calendar event with sufficient confidence
        if (
            not initial_extraction.is_calendar_event 
            or initial_extraction.confidence_score < confidence_threshold
        ): 
            logger.warning(
                f"[{request_id}] Gate check failed: is_calendar_event={initial_extraction.is_calendar_event}, confidence_score={initial_extraction.confidence_score:.2f}"
            )
            return None
        
        logger.info(f"[{request_id}] Gate check passed, proceeding to parse event details")

        # Step 2: Second LLM call: Parse event details
        event_details = parse_event_details(initial_extraction.description)

        # Step 3: Third LLM call: Generate confirmation
        confirmation = generate_confirmation(event_details)
        logger.info(f"[{request_id}] Calendar request processing completed successfully")

        return confirmation
    
    except Exception as e:
        logger.error(f"[{request_id}] Error processing calendar request: {str(e)}")
        return None

# --------------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------------

def valid_input():
    user_input = "Let's schedule a 1h team meeting next Friday at 7pm with Nyasa and Kiara to discuss the project roadmap."

    result = process_calendar_request(user_input)
    if result:
        print(format_event_confirmation(result))
    else:
        print("This doesn't appear to be a calendar event request.")

# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

def invalid_input():
    invalid_input = "Can you send an email to Ayaan and Bob to discuss the project roadmap?"

    result = process_calendar_request(invalid_input)
    if result:
        print(format_event_confirmation(result))
    else:
        print("This doesn't appear to be a calendar event request.")

if __name__ == "__main__":
    valid_input()
    # invalid_input()