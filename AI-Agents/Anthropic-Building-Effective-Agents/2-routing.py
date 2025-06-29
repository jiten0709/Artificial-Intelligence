import os
from dotenv import load_dotenv
import sys
from typing import Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, ValidationError
import openai
from openai import OpenAIError
import logging
import json
from datetime import datetime, timedelta
import dateutil.parser
import pytz
from functools import wraps
import time
from enum import Enum
import random

# --------------------------------------------------------------
# Step 1: Configuration Management
# --------------------------------------------------------------

class Config(BaseModel):
    """Configuration for the calendar request processor"""
    openai_api_key: str
    model: str = "gpt-4o-mini"
    api_timeout: int = 30
    max_retries: int = 2
    retry_delay: int = 1
    confidence_threshold: float = 0.7
    log_level: str = "INFO"
    timezone: str = "UTC"
    default_duration_minutes: int = 60
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        load_dotenv()
        
        # Required values
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Optional values with defaults
        return cls(
            openai_api_key=openai_api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_timeout=int(os.getenv("API_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            retry_delay=int(os.getenv("RETRY_DELAY", "2")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            timezone=os.getenv("TIMEZONE", "UTC"),
            default_duration_minutes=int(os.getenv("DEFAULT_DURATION_MINUTES", "60")),
        )

# Load configuration
try:
    config = Config.from_env()
except ValueError as e:
    print(f"Configuration error: {e}", file=sys.stderr)
    sys.exit(1)

# Set up logging configuration
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

logging.basicConfig(
    level=log_levels.get(config.log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=config.openai_api_key)

# --------------------------------------------------------------
# Step 2: Error Handling
# --------------------------------------------------------------

class ProcessingError(Exception):
    """Base exception for calendar processing errors"""
    pass

class APIError(ProcessingError):
    """Error communicating with the OpenAI API"""
    pass

class ParseError(ProcessingError):
    """Error parsing response from the API"""
    pass

class ValidationError(ProcessingError):
    """Error validating input or output data"""
    pass

class OutputFormat(str, Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    DETAILED = "detailed"

def retry_on_error(max_retries=None, retry_delay=None):
    """Decorator to retry a function on failure with exponential backoff for rate limits"""
    max_retries = max_retries or config.max_retries
    base_delay = retry_delay or config.retry_delay
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except OpenAIError as e:
                    # Check if it's a rate limit error
                    is_rate_limit = hasattr(e, 'status_code') and e.status_code == 429
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff for rate limits
                        if is_rate_limit:
                            # Exponential backoff with jitter for rate limits
                            delay = min(60, base_delay * (2 ** attempt) + random.uniform(0, 1))
                            logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f}s ({attempt+1}/{max_retries})")
                        else:
                            delay = base_delay
                            logger.warning(f"API error: {e}. Retrying in {delay}s ({attempt+1}/{max_retries})")
                        
                        time.sleep(delay)
                    else:
                        if is_rate_limit:
                            logger.error(f"Rate limit exceeded after {max_retries} retries. Try again later.")
                            raise APIError("OpenAI API rate limit exceeded. Please try again after a few minutes.")
                        else:
                            logger.error(f"API error after {max_retries} retries: {e}")
                            raise APIError(f"Failed to communicate with AI service: {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    raise ParseError(f"Failed to parse AI response: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise
        return wrapper
    return decorator

# --------------------------------------------------------------
# Step 3: Data Models with Improved Date Handling
# --------------------------------------------------------------

class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""
    
    request_type: Literal['new_event', 'modify_event', 'other'] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(
        description="Confidence score between 0 and 1"
    )
    cleaned_description: str = Field(
        description="Cleaned description of the request"
    )

class NewEventDetails(BaseModel):
    """Details for creating a new event"""

    name: str = Field(
        description="Name of the event"
    )
    date: str = Field(
        description="Date and time of the event (ISO 8601)"
    )
    duration_minutes: int = Field(
        description="Duration in minutes", default=60
    )
    participants: list[str] = Field(
        description="List of participants"
    )
    
    def get_parsed_date(self):
        """Parse the date string into a datetime object"""
        try:
            date_obj = dateutil.parser.parse(self.date)
            if date_obj.tzinfo is None:
                # Assume local timezone if not specified
                local_tz = pytz.timezone(config.timezone)
                date_obj = local_tz.localize(date_obj)
            return date_obj
        except Exception as e:
            raise ValidationError(f"Invalid date format: {e}")
    
    def formatted_date(self, format_str="%Y-%m-%d %H:%M %Z"):
        """Return a formatted date string"""
        return self.get_parsed_date().strftime(format_str)
    
    def is_future_date(self):
        """Check if the event date is in the future"""
        return self.get_parsed_date() > datetime.now(pytz.timezone(config.timezone))

class Change(BaseModel):
    """Details for changing an existing event"""

    field: str = Field(
        description="Field to change"
    )
    new_value: str = Field(
        description="New value for the field"
    )

class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""

    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(
        description="List of changes to make"
    )
    participants_to_add: list[str] = Field(
        description="New participants to add", default=[]
    )
    participants_to_remove: list[str] = Field(
        description="Participants to remove", default=[]
    )
    
    def get_date_change(self):
        """Extract date change if present"""
        for change in self.changes:
            if change.field.lower() in ["date", "time", "datetime"]:
                try:
                    return dateutil.parser.parse(change.new_value)
                except Exception:
                    return None
        return None
    
    def formatted_changes(self):
        """Return formatted changes for display"""
        result = []
        for change in self.changes:
            if change.field.lower() in ["date", "time", "datetime"]:
                try:
                    date_obj = dateutil.parser.parse(change.new_value)
                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    result.append(f"{change.field}={formatted_date}")
                except Exception:
                    result.append(f"{change.field}={change.new_value}")
            else:
                result.append(f"{change.field}={change.new_value}")
        return result

class CalendarResponse(BaseModel): 
    """Final response format"""

    success: bool = Field(
        description="Whether the operation was successful"
    )
    message: str = Field(
        description="User-friendly response message"
    )
    calendar_link: Optional[str] = Field(
        description="Calendar link if applicable", default=None
    )
    details: Optional[Dict[str, Any]] = Field(
        description="Additional details for the response", default=None
    )
    error: Optional[str] = Field(
        description="Error message if operation failed", default=None
    )
    
    def format_output(self, format_type: OutputFormat = OutputFormat.TEXT):
        """Format the response based on the requested output format"""
        if format_type == OutputFormat.JSON:
            return json.dumps(self.dict(), indent=2)
        elif format_type == OutputFormat.DETAILED:
            if self.success:
                result = f"✅ {self.message}\n"
                if self.calendar_link:
                    result += f"📅 Link: {self.calendar_link}\n"
                if self.details:
                    result += "📋 Details:\n"
                    for k, v in self.details.items():
                        result += f"  - {k}: {v}\n"
                return result
            else:
                result = f"❌ {self.message}\n"
                if self.error:
                    result += f"🚫 Error: {self.error}\n"
                return result
        else:  # OutputFormat.TEXT (default)
            return self.message

# --------------------------------------------------------------
# Step 4: Processing Functions with Better Error Handling
# --------------------------------------------------------------

@retry_on_error()
def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("Routing calendar request")

    completion = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event or modify an existing one. Respond as a JSON object with keys: request_type, confidence_score, cleaned_description. IMPORTANT: request_type MUST be exactly one of these values: 'new_event', 'modify_event', or 'other'.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        timeout=config.api_timeout
    )
    
    response_content = completion.choices[0].message.content
    try:
        # Parse the raw JSON first
        parsed_json = json.loads(response_content)
        
        # Normalize the request_type to match our expected values
        request_type_mapping = {
            "create": "new_event",
            "create_new_event": "new_event",
            "new": "new_event",
            "modify": "modify_event",
            "modify_existing_event": "modify_event",
            "update": "modify_event",
            "unknown": "other",
        }
        
        if parsed_json.get("request_type") in request_type_mapping:
            parsed_json["request_type"] = request_type_mapping[parsed_json["request_type"]]
            
        result = CalendarRequestType(**parsed_json)
        logger.info(
            f"Request routed as: {result.request_type} with confidence: {result.confidence_score}"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to parse routing response: {e}")
        logger.debug(f"Raw response: {response_content}")
        raise ParseError(f"Failed to understand the request type: {e}")

@retry_on_error()
def handle_new_event(description: str) -> CalendarResponse:
    """Process a new event request"""
    logger.info("Processing new event request")

    today = datetime.now(pytz.timezone(config.timezone))
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"

    # Get event details
    completion = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Extract details for creating a new calendar event. Respond as a JSON object with keys: name, date, duration_minutes, participants. Format the date in ISO 8601 format.",
            },
            {"role": "user", "content": description},
        ],
        response_format={"type": "json_object"},
        timeout=config.api_timeout
    )

    response_content = completion.choices[0].message.content
    try:
        details = NewEventDetails(**json.loads(response_content))
        logger.info(f"New event details extracted: {details}")
        
        # Validate date is in the future
        if not details.is_future_date():
            logger.warning(f"Event date is in the past: {details.date}")
            return CalendarResponse(
                success=False,
                message="Cannot create an event in the past. Please specify a future date and time.",
                error="Past date specified"
            )
            
        # Generate response with additional details
        return CalendarResponse(
            success=True,
            message=f"Created new event '{details.name}' for {details.formatted_date()} with {', '.join(details.participants)}",
            calendar_link=f"calendar://new?event={details.name}",
            details={
                "name": details.name,
                "date": details.formatted_date(),
                "duration": f"{details.duration_minutes} minutes",
                "participants": details.participants
            }
        )
    except Exception as e:
        logger.error(f"Failed to parse new event details: {e}")
        logger.debug(f"Raw response: {response_content}")
        return CalendarResponse(
            success=False,
            message="Failed to create the event due to invalid details.",
            error=f"Failed to parse event details: {str(e)}"
        )

@retry_on_error()
def handle_modify_event(description: str) -> CalendarResponse:
    """Process an event modification request"""
    logger.info("Processing event modification request")

    today = datetime.now(pytz.timezone(config.timezone))
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}"

    # Get modification details
    completion = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Extract details for modifying an existing calendar event. Respond as a JSON object with keys: event_identifier, changes (which should be a list of objects with 'field' and 'new_value' properties), participants_to_add, participants_to_remove. Format any dates in ISO 8601 format.",
            },
            {"role": "user", "content": description},
        ],
        response_format={"type": "json_object"},
        timeout=config.api_timeout
    )

    response_content = completion.choices[0].message.content
    try:
        # Parse the raw JSON first
        parsed_json = json.loads(response_content)
        
        # Check if changes is a dictionary and convert it to a list of Change objects
        if "changes" in parsed_json and isinstance(parsed_json["changes"], dict):
            changes_list = []
            for field, new_value in parsed_json["changes"].items():
                changes_list.append({"field": field, "new_value": str(new_value)})
            parsed_json["changes"] = changes_list
        
        details = ModifyEventDetails(**parsed_json)
        logger.info(f"Modification details extracted: {details}")
        
        # Validate date changes if present
        date_change = details.get_date_change()
        if date_change and date_change < today:
            logger.warning(f"Modified date is in the past: {date_change}")
            return CalendarResponse(
                success=False,
                message="Cannot modify an event to a past date. Please specify a future date and time.",
                error="Past date specified"
            )
            
        # Generate response with additional details
        additional_details = {
            "event": details.event_identifier,
            "changes": details.formatted_changes(),
        }
        
        if details.participants_to_add:
            additional_details["added_participants"] = details.participants_to_add
            
        if details.participants_to_remove:
            additional_details["removed_participants"] = details.participants_to_remove
            
        return CalendarResponse(
            success=True,
            message=f"Modified event '{details.event_identifier}' with changes: {', '.join(details.formatted_changes())}",
            calendar_link=f"calendar://modify?event={details.event_identifier}",
            details=additional_details
        )
    except Exception as e:
        logger.error(f"Failed to parse modification details: {e}")
        logger.debug(f"Raw response: {response_content}")
        return CalendarResponse(
            success=False,
            message="Failed to modify the event due to invalid details.",
            error=f"Failed to parse modification details: {str(e)}"
        )

def process_calendar_request(user_input: str, output_format: OutputFormat = OutputFormat.TEXT) -> Union[CalendarResponse, str]:
    """Main function to process calendar requests"""
    logger.info(f"Received user input: {user_input}")
    
    if not user_input or user_input.strip() == "":
        response = CalendarResponse(
            success=False,
            message="Please provide a calendar request.",
            error="Empty input"
        )
        return response if output_format == OutputFormat.JSON else response.format_output(output_format)

    try:
        # Step 1: Route the request
        route_result = route_calendar_request(user_input)

        # Step 2: Check confidence 
        if route_result.confidence_score < config.confidence_threshold:
            logger.warning(f"Low confidence in request type: {route_result.confidence_score}")
            response = CalendarResponse(
                success=False,
                message="I'm not sure how to handle that request. Please try again with more details.",
                error=f"Low confidence score: {route_result.confidence_score}"
            )
            return response if output_format == OutputFormat.JSON else response.format_output(output_format)
        
        # Step 3: Route to appropriate handler
        if route_result.request_type == "new_event":
            response = handle_new_event(route_result.cleaned_description)
        elif route_result.request_type == "modify_event":
            response = handle_modify_event(route_result.cleaned_description)
        else:
            logger.warning(f"Unknown request type: {route_result.request_type}")
            response = CalendarResponse(
                success=False,
                message="I'm not sure how to handle that request. Please try again with a calendar-related request.",
                error=f"Unsupported request type: {route_result.request_type}"
            )
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        response = CalendarResponse(
            success=False,
            message=f"Sorry, I encountered an error while processing your request: {str(e)}",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        response = CalendarResponse(
            success=False,
            message="Sorry, something went wrong while processing your request.",
            error=f"Unexpected error: {str(e)}"
        )
        
    return response if output_format == OutputFormat.JSON else response.format_output(output_format)

# --------------------------------------------------------------
# Step 5: Testing with Different Inputs
# --------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Testing New Event Creation ===")
    new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Nyasa and Kiara"
    result = process_calendar_request(new_event_input, OutputFormat.DETAILED)
    print(result)

    print("\n=== Testing Event Modification ===")
    modify_event_input = "Can you move the team meeting with Nyasa and Kiara to Wednesday at 3pm instead and add Jitu to the meeting?"
    result = process_calendar_request(modify_event_input, OutputFormat.DETAILED)
    print(result)

    print("\n=== Testing Invalid Request ===")
    invalid_input = "What's the weather like today?"
    result = process_calendar_request(invalid_input, OutputFormat.DETAILED)
    print(result)
