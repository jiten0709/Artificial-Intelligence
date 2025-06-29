import os 
from dotenv import load_dotenv
load_dotenv()

import logging
import asyncio
import random
import json
import time
from typing import List, Dict, Any, Optional
from functools import wraps
from enum import Enum

import openai
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Constants
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 60

# --------------------------------------------------------------
# Step 1: Rate Limiter and Retry Logic
# --------------------------------------------------------------

class RateLimiter:
    """Rate limiter to prevent too many API requests"""
    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock() if asyncio.get_event_loop_policy().get_event_loop().is_running() else None
    
    async def wait_if_needed_async(self):
        """Async version of rate limiting"""
        if not self.lock:
            self.lock = asyncio.Lock()
            
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                # Wait until we can make another request
                oldest_request = min(self.request_times)
                wait_time = 60 - (now - oldest_request) + 0.1  # Add a small buffer
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(time.time())
    
    def wait_if_needed(self):
        """Synchronous version of rate limiting"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until we can make another request
            oldest_request = min(self.request_times)
            wait_time = 60 - (now - oldest_request) + 0.1  # Add a small buffer
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Record this request
        self.request_times.append(time.time())

# Create a global rate limiter
rate_limiter = RateLimiter(requests_per_minute=15)  # Conservative limit

def retry_with_exponential_backoff(
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    backoff_factor: float = 2.0,
    error_types: tuple = (openai.RateLimitError, openai.APIError)
):
    """Decorator for retrying function calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = initial_delay
            
            while True:
                try:
                    # Apply rate limiting
                    rate_limiter.wait_if_needed()
                    
                    # Call the function
                    return func(*args, **kwargs)
                
                except error_types as e:
                    retry_count += 1
                    
                    # Check if we've exceeded the max retries
                    if retry_count > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded.")
                        raise
                    
                    # Calculate the next delay with jitter
                    delay = min(max_delay, delay * backoff_factor) * (0.5 + random.random())
                    
                    logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}. Waiting {delay:.2f}s...")
                    time.sleep(delay)
                
                except Exception as e:
                    # For other exceptions, don't retry
                    logger.error(f"Non-retryable error: {str(e)}")
                    raise
                    
        return wrapper
    return decorator

# --------------------------------------------------------------
# Step 2: Define the data models
# --------------------------------------------------------------

class DataQuality(Enum):
    """Enum representing data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EvaluationCriterion(BaseModel):
    """Evaluation criterion for the generated content"""
    name: str = Field(description="Name of the evaluation criterion")
    description: str = Field(description="Detailed description of what the criterion measures")
    score: float = Field(description="Score for the criterion from 1-10, where 1 is the worst and 10 is the best")
    feedback: str = Field(description="Specific feedback for this criterion")

class ContentEvaluation(BaseModel):
    """Complete evaluation of the generated content"""
    overall_score: float = Field(description="Overall score for the content from 1-10, where 1 is the worst and 10 is the best")
    criteria_scores: List[EvaluationCriterion] = Field(description="List of detailed scores for each evaluation criterion")
    strengths: List[str] = Field(description="List the key strengths of the content")
    weaknesses: List[str] = Field(description="List the key weaknesses of the content")
    improvement_suggestions: List[str] = Field(description="List of specific suggestions for improvement")

class OptimizedContent(BaseModel):
    """Optimized content based on evaluation feedback"""
    content: str = Field(description="The optimized content after applying feedback")
    improvements_made: List[str] = Field(description="List of specific improvements made to the original content based on feedback")
    expected_score_increase: float = Field(description="Expected increase in score compared to the original content")

class WorkflowConfig(BaseModel):
    """Configuration for the evaluator-optimizer workflow"""
    dataset_description: str = Field(description="Description of the dataset being analyzed")
    data_quality: DataQuality = Field(description="Quality level of the dataset")
    analysis_goal: str = Field(description="Goal of the analysis")
    target_audience: str = Field(description="Target audience for the report")
    required_length: int = Field(default=500, description="Required length of the report in words")
    focus_area: str = Field(default="actionable and insightful", description="Focus area for the analysis")
    output_dir: Optional[str] = Field(default="output", description="Directory to save output files")
    save_results: bool = Field(default=True, description="Whether to save results to files")
    
# --------------------------------------------------------------
# Step 3: Define the prompts
# --------------------------------------------------------------

DATA_GENERATION_PROMPT = """
Create a data analysis report based on the following dataset and requirements:

Dataset: {dataset_description}
Data Quality: {data_quality}
Analysis Goal: {analysis_goal}
Target Audience: {target_audience}
Required Length: {required_length} words

Your report should include:
1. Executive summary
2. Key findings
3. Methodology
4. Detailed analysis
5. Conclusions and recommendations

Focus on making the analysis {focus_area}.
"""

EVALUATION_PROMPT = """
Evaluate the following data analysis report based on these criteria:
1. Accuracy (Is the analysis factually correct and well-reasoned?)
2. Clarity (Is the report clear and easy to understand?)
3. Relevance (Does it address the analysis goal?)
4. Methodology (Is the analytical approach sound?)
5. Insights (Does it provide valuable insights?)
6. Structure (Is it well-organized?)
7. Actionability (Are the recommendations practical?)

Report:
{content}

Dataset Info: {dataset_description}
Data Quality: {data_quality}
Analysis Goal: {analysis_goal}
Target Audience: {target_audience}

Provide a detailed evaluation with specific examples from the report. Be critical but fair.
Suggest specific improvements for each area that needs work.

Return your response in JSON format.
"""

OPTIMIZATION_PROMPT = """
Optimize the following data analysis report based on the evaluation feedback:

Original Report:
{original_content}

Evaluation:
{evaluation}

Dataset Info: {dataset_description}
Data Quality: {data_quality}
Analysis Goal: {analysis_goal}
Target Audience: {target_audience}

Create an improved version that addresses all the weaknesses identified in the evaluation.
Make sure to maintain the strengths of the original while fixing its issues.
The improved report should still follow the required structure:
1. Executive summary
2. Key findings
3. Methodology
4. Detailed analysis
5. Conclusions and recommendations
"""

OPTIMIZATION_METADATA_PROMPT = """
Compare the original content with the optimized version and identify:
1. The specific improvements made
2. The expected increase in score

Original content:
{original_content}

Optimized content:
{optimized_content}

Original evaluation:
{evaluation}

Return your response in JSON format.
"""

# --------------------------------------------------------------
# Step 4: API Interaction Functions
# --------------------------------------------------------------

@retry_with_exponential_backoff()
def make_openai_call(messages: List[Dict[str, str]], response_format: Optional[Dict] = None, 
                     tools: Optional[List[Dict]] = None, tool_choice: Optional[Dict] = None) -> Any:
    """Make a call to the OpenAI API with retries and rate limiting"""
    try:
        completion_args = {
            "model": model,
            "messages": messages
        }
        
        if response_format:
            completion_args["response_format"] = response_format
        
        if tools:
            completion_args["tools"] = tools
            
        if tool_choice:
            completion_args["tool_choice"] = tool_choice
            
        completion = client.chat.completions.create(**completion_args)
        return completion
    except Exception as e:
        logger.error(f"Error making OpenAI call: {e}")
        raise

# --------------------------------------------------------------
# Step 5: Implement the enhanced evaluator-optimizer workflow
# --------------------------------------------------------------

class EvaluatorOptimizer:
    """Enhanced evaluator-optimizer workflow with error handling and performance optimizations"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.original_content = ""
        self.evaluation = None
        self.optimized_content = None
        self.config = config
        self.cache = {}  # Simple cache to avoid redundant API calls
        
        # Create output directory if needed
        if config and config.save_results and config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)

    @retry_with_exponential_backoff()
    def generate_content(self, 
                        dataset_description: str, 
                        data_quality: DataQuality, 
                        analysis_goal: str, 
                        target_audience: str,
                        required_length: int = 500,
                        focus_area: str = "actionable and insightful") -> str:
        """Generate the initial content based on the provided dataset and input requirements."""
        try:
            # Create a cache key for this request
            cache_key = f"generate_{dataset_description}_{data_quality.value}_{analysis_goal}_{required_length}_{focus_area}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info("Using cached content generation result")
                self.original_content = self.cache[cache_key]
                return self.original_content
            
            logger.info(f"Generating initial content for analysis goal: {analysis_goal}")
            
            messages = [
                {
                    "role": "system",
                    "content": DATA_GENERATION_PROMPT.format(
                        dataset_description=dataset_description,
                        data_quality=data_quality.value,
                        analysis_goal=analysis_goal,
                        target_audience=target_audience,
                        required_length=required_length,
                        focus_area=focus_area
                    )
                }
            ]
            
            completion = make_openai_call(messages)
            self.original_content = completion.choices[0].message.content
            
            # Cache the result
            self.cache[cache_key] = self.original_content
            
            logger.info(f"Content generated having {len(self.original_content)} characters")
            return self.original_content
        
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    @retry_with_exponential_backoff()
    def evaluate_content(self,
                        dataset_description: str, 
                        data_quality: DataQuality, 
                        analysis_goal: str, 
                        target_audience: str) -> ContentEvaluation:
        """Evaluate the generated content based on predefined criteria."""
        try:
            if not self.original_content:
                raise ValueError("No content to evaluate. Please generate content first.")
            
            # Create a cache key for this request
            cache_key = f"evaluate_{hash(self.original_content)}_{data_quality.value}_{analysis_goal}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info("Using cached evaluation result")
                self.evaluation = self.cache[cache_key]
                return self.evaluation
            
            logger.info("Evaluating content...")
            
            messages = [
                {
                    "role": "system",
                    "content": EVALUATION_PROMPT.format(
                        content=self.original_content,
                        dataset_description=dataset_description,
                        data_quality=data_quality.value,
                        analysis_goal=analysis_goal,
                        target_audience=target_audience
                    )
                }
            ]
            
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "evaluate_content",
                        "description": "Evaluate the quality of the content",
                        "parameters": ContentEvaluation.model_json_schema()                    
                    }
                }
            ]
            
            tool_choice = {"type": "function", "function": {"name": "evaluate_content"}}
            response_format = {"type": "json_object"}
            
            completion = make_openai_call(
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice
            )
            
            tool_call = completion.choices[0].message.tool_calls[0]
            self.evaluation = ContentEvaluation.model_validate_json(tool_call.function.arguments)
            
            # Cache the result
            self.cache[cache_key] = self.evaluation
            
            logger.info(f"Content evaluated with overall score: {self.evaluation.overall_score}")
            return self.evaluation
        
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            raise

    @retry_with_exponential_backoff()
    def optimize_content(self,
                        dataset_description: str, 
                        data_quality: DataQuality, 
                        analysis_goal: str, 
                        target_audience: str) -> OptimizedContent:
        """Optimize the content based on the evaluation feedback."""
        try:
            if not self.original_content:
                raise ValueError("No content to optimize. Please generate content first.")
            
            if not self.evaluation:
                raise ValueError("No evaluation available. Please evaluate content first.")
            
            # Create a cache key for this request
            cache_key = f"optimize_{hash(self.original_content)}_{hash(str(self.evaluation))}_{data_quality.value}_{analysis_goal}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info("Using cached optimization result")
                self.optimized_content = self.cache[cache_key]
                return self.optimized_content
            
            logger.info("Optimizing content based on evaluation feedback...")

            # First generate the optimized content
            messages_content = [
                {
                    "role": "system",
                    "content": OPTIMIZATION_PROMPT.format(
                        original_content=self.original_content,
                        evaluation=self.evaluation.model_dump_json(indent=2),
                        dataset_description=dataset_description,
                        data_quality=data_quality.value,
                        analysis_goal=analysis_goal,
                        target_audience=target_audience
                    )
                }
            ]
            
            completion_content = make_openai_call(messages=messages_content)
            optimized_text = completion_content.choices[0].message.content
            
            # Then get metadata about the optimized content
            messages_metadata = [
                {
                    "role": "system",
                    "content": OPTIMIZATION_METADATA_PROMPT.format(
                        original_content=self.original_content,
                        optimized_content=optimized_text,
                        evaluation=self.evaluation.model_dump_json(indent=2)
                    ) + "\n\nPlease provide your response in JSON format."
                }
            ]
            
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "optimize_content",
                        "description": "Provide metadata about the optimization",
                        "parameters": OptimizedContent.model_json_schema()
                    }
                }
            ]
            
            tool_choice = {"type": "function", "function": {"name": "optimize_content"}}
            response_format = {"type": "json_object"}
            
            completion_metadata = make_openai_call(
                messages=messages_metadata,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice
            )
            
            tool_call = completion_metadata.choices[0].message.tool_calls[0]
            metadata = json.loads(tool_call.function.arguments)

            # Create the OptimizedContent object with the optimized text and metadata
            self.optimized_content = OptimizedContent(
                content=optimized_text,
                improvements_made=metadata["improvements_made"],
                expected_score_increase=metadata["expected_score_increase"]
            )
            
            # Cache the result
            self.cache[cache_key] = self.optimized_content
            
            logger.info(f"Content optimized with expected score increase: {self.optimized_content.expected_score_increase}")
            return self.optimized_content

        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            raise

    def save_results(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
        """Save the workflow results to files"""
        try:
            if not output_dir:
                output_dir = self.config.output_dir if self.config else "output"
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original content
            with open(os.path.join(output_dir, "original_content.txt"), "w") as f:
                f.write(results['original_content'])
            
            # Save evaluation
            with open(os.path.join(output_dir, "evaluation.txt"), "w") as f:
                f.write(f"Overall Score: {results['evaluation'].overall_score}/10\n\n")
                f.write("Criteria Scores:\n")
                for criterion in results['evaluation'].criteria_scores:
                    f.write(f"- {criterion.name}: {criterion.score}/10\n")
                    f.write(f"  {criterion.feedback}\n\n")
                f.write("\nStrengths:\n")
                for strength in results['evaluation'].strengths:
                    f.write(f"- {strength}\n")
                f.write("\nWeaknesses:\n")
                for weakness in results['evaluation'].weaknesses:
                    f.write(f"- {weakness}\n")
                f.write("\nImprovement Suggestions:\n")
                for suggestion in results['evaluation'].improvement_suggestions:
                    f.write(f"- {suggestion}\n")
            
            # Save optimized content
            with open(os.path.join(output_dir, "optimized_content.txt"), "w") as f:
                f.write(results['optimized_content'].content)
            
            # Save metadata as JSON
            with open(os.path.join(output_dir, "results_metadata.json"), "w") as f:
                json.dump({
                    "overall_score": results['evaluation'].overall_score,
                    "strengths": results['evaluation'].strengths,
                    "weaknesses": results['evaluation'].weaknesses,
                    "improvements_made": results['optimized_content'].improvements_made,
                    "expected_score_increase": results['optimized_content'].expected_score_increase
                }, f, indent=2)
                
            logger.info(f"Results saved to directory: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def run_workflow(self,
                    dataset_description: Optional[str] = None,
                    data_quality: Optional[DataQuality] = None,
                    analysis_goal: Optional[str] = None,
                    target_audience: Optional[str] = None,
                    required_length: Optional[int] = None,
                    focus_area: Optional[str] = None,
                    save_results: Optional[bool] = None) -> Dict[str, Any]:
        """Run the complete evaluator-optimizer workflow."""
        start_time = time.time()
        
        # Use parameters from config if not explicitly provided
        if self.config:
            dataset_description = dataset_description or self.config.dataset_description
            data_quality = data_quality or self.config.data_quality
            analysis_goal = analysis_goal or self.config.analysis_goal
            target_audience = target_audience or self.config.target_audience
            required_length = required_length or self.config.required_length
            focus_area = focus_area or self.config.focus_area
            
            if save_results is None:
                save_results = self.config.save_results
        
        if not all([dataset_description, data_quality, analysis_goal, target_audience]):
            raise ValueError("Missing required parameters for workflow. Provide them directly or via config.")
            
        required_length = required_length or 500
        focus_area = focus_area or "actionable and insightful"
        
        logger.info(f"Starting evaluator-optimizer workflow for {analysis_goal}...")

        try:
            # Step 1: Generate initial content
            original_content = self.generate_content(
                dataset_description=dataset_description,
                data_quality=data_quality,
                analysis_goal=analysis_goal,
                target_audience=target_audience,
                required_length=required_length,
                focus_area=focus_area
            )
            
            # Step 2: Evaluate the generated content
            evaluation = self.evaluate_content(
                dataset_description=dataset_description,
                data_quality=data_quality,
                analysis_goal=analysis_goal,
                target_audience=target_audience
            )

            # Step 3: Optimize the content based on evaluation feedback
            optimized_content = self.optimize_content(
                dataset_description=dataset_description,
                data_quality=data_quality,
                analysis_goal=analysis_goal,
                target_audience=target_audience
            )

            # Prepare results
            results = {
                "original_content": original_content,
                "evaluation": evaluation,
                "optimized_content": optimized_content
            }
            
            # Save results if requested
            if save_results:
                self.save_results(results)

            end_time = time.time()
            logger.info(f"Workflow completed in {end_time - start_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            raise

# --------------------------------------------------------------
# Step 6: Command-line interface and example usage
# --------------------------------------------------------------

def print_results(results: Dict[str, Any]) -> None:
    """Print workflow results in a formatted way"""
    print("\n\n=== EVALUATION RESULTS ===\n")
    print(f"Overall Score: {results['evaluation'].overall_score}/10")
    
    print("\nStrengths:")
    for strength in results['evaluation'].strengths:
        print(f"- {strength}")
    
    print("\nWeaknesses:")
    for weakness in results['evaluation'].weaknesses:
        print(f"- {weakness}")
    
    print("\n\n=== IMPROVEMENTS MADE ===\n")
    for improvement in results['optimized_content'].improvements_made:
        print(f"- {improvement}")
    
    print(f"\nExpected Score Increase: {results['optimized_content'].expected_score_increase} points")
    
    print("\n\n=== OPTIMIZED CONTENT ===\n")
    print(results['optimized_content'].content)

def print_results(results: Dict[str, Any], output_file: Optional[str] = None) -> None:
    """Print workflow results in a formatted way and optionally save to a well-structured file"""
    # Create the output text with proper formatting and structure
    output_text = "=" * 80 + "\n"
    output_text += "CONTENT EVALUATION AND OPTIMIZATION REPORT\n"
    output_text += "=" * 80 + "\n\n"
    
    # SECTION 1: Evaluation Results
    output_text += "I. EVALUATION RESULTS\n"
    output_text += "-" * 80 + "\n\n"
    output_text += f"Overall Score: {results['evaluation'].overall_score}/10\n\n"
    
    output_text += "A. Detailed Criteria Scores:\n"
    for criterion in results['evaluation'].criteria_scores:
        output_text += f"   • {criterion.name}: {criterion.score}/10\n"
        output_text += f"     {criterion.feedback}\n\n"
    
    output_text += "B. Strengths:\n"
    for i, strength in enumerate(results['evaluation'].strengths, 1):
        output_text += f"   {i}. {strength}\n"
    
    output_text += "\nC. Weaknesses:\n"
    for i, weakness in enumerate(results['evaluation'].weaknesses, 1):
        output_text += f"   {i}. {weakness}\n"
    
    output_text += "\nD. Improvement Suggestions:\n"
    for i, suggestion in enumerate(results['evaluation'].improvement_suggestions, 1):
        output_text += f"   {i}. {suggestion}\n"
    
    # SECTION 2: Optimization Results
    output_text += "\n\nII. OPTIMIZATION RESULTS\n"
    output_text += "-" * 80 + "\n\n"
    output_text += f"Expected Score Increase: {results['optimized_content'].expected_score_increase} points\n\n"
    
    output_text += "A. Improvements Made:\n"
    for i, improvement in enumerate(results['optimized_content'].improvements_made, 1):
        output_text += f"   {i}. {improvement}\n"
    
    # SECTION 3: Optimized Content
    output_text += "\n\nIII. OPTIMIZED CONTENT\n"
    output_text += "-" * 80 + "\n\n"
    output_text += results['optimized_content'].content
    
    # Print to console
    print(output_text)
    
    # Save to file if requested
    if output_file:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(output_text)
            logger.info(f"Detailed report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report to file: {e}")

if __name__ == "__main__":
    try:
        # Example configuration
        config = WorkflowConfig(
            dataset_description="Retail sales data for a clothing company over the past 3 years, including customer demographics, purchase history, and seasonal trends.",
            data_quality=DataQuality.MEDIUM,
            analysis_goal="Identify opportunities to increase customer retention and lifetime value",
            target_audience="Marketing team and senior management",
            required_length=800,
            focus_area="data-driven and actionable",
            output_dir="AI-Agents/patterns/workflows/2-workflow-patterns/output",
            save_results=True
        )
        
        # Initialize the workflow with configuration
        workflow = EvaluatorOptimizer(config=config)
        
        # Run the workflow
        result = workflow.run_workflow()
        
        # Print the results and save to a file
        summary_file = os.path.join(config.output_dir, "detailed_report.txt")
        print_results(result, output_file=summary_file)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise