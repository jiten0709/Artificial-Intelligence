import os 
from dotenv import load_dotenv
load_dotenv()

import logging
from typing import List, Dict
from pydantic import BaseModel, Field
import json
import openai
from enum import Enum
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# constants
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

# --------------------------------------------------------------
# Step 1: Define the data models
# --------------------------------------------------------------

class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EvaluationCriterion(BaseModel):
    """evaluation criterion for the generated content"""
    name: str = Field(description="Name of the evaluation criterion")
    description: str = Field(description="detailed description of what the criterion measures")
    score: float = Field(description="Score for the criterion from 1-10, where 1 is the worst and 10 is the best")
    feedback: str = Field(description="specific feedback for this criterion")

class ContentEvaluation(BaseModel):
    """complete evaluation of the generated content"""
    overall_score: float = Field(description="Overall score for the content from 1-10, where 1 is the worst and 10 is the best")
    criteria_scores: List[EvaluationCriterion] = Field(description="List of detailed scores for each evaluation criterion")
    strengths: List[str] = Field(description="List the key strengths of the content")
    weaknesses: List[str] = Field(description="List the key weaknesses of the content")
    improvement_suggestions: List[str] = Field(description="List of specific suggestions for improvement")

class OptimizedContent(BaseModel):
    """optimized content based on evaluation feedback"""
    content: str = Field(description="The optimized content after applying feedback")
    improvements_made: List[str] = Field(description="List of specific improvements made to the original content based on feedback")
    expected_score_increase: float = Field(description="Expected increase in score compared to the original content")

# --------------------------------------------------------------
# Step 2: Define the prompts
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
"""

# --------------------------------------------------------------
# Step 3: Implement the evaluator-optimizer workflow
# --------------------------------------------------------------

class EvaluatorOptimizer:
    def __init__(self):
        self.original_content = ""
        self.evaluation = None
        self.optimized_content = None

    def generate_content(self,
                        dataset_description: str, 
                        data_quality: DataQuality, 
                        analysis_goal: str, 
                        target_audience: str,
                        required_length: int = 500,
                        focus_area: str = "actionable and insightful") -> str:
        """Generate the initial content based on the provided dataset and input requirements."""
        try:
            logger.info(f"Generating initial content for analysis goal: {analysis_goal}")

            completion = client.chat.completions.create(
                model=model,
                messages=[
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
            )

            self.original_content = completion.choices[0].message.content
            logger.info(f"content generated having {len(self.original_content)} characters")

            return self.original_content
        
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    def evaluate_content(self,
                        dataset_description: str, 
                        data_quality: DataQuality, 
                        analysis_goal: str, 
                        target_audience: str) -> ContentEvaluation:
        """Evaluate the generated content based on predefined criteria."""
        try:
            if not self.original_content:
                raise ValueError("No content to evaluate. Please generate content first.")
            
            logger.info("Evaluating content...")

            completion = client.chat.completions.create(
                model=model,
                messages=[
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
                ],
                response_format={"type": "json_object"},
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "evaluate_content",
                            "description": "Evaluate the quality of the content",
                            "parameters": ContentEvaluation.model_json_schema()                    
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "evaluate_content"}}
            )

            tool_call = completion.choices[0].message.tool_calls[0]
            self.evaluation = ContentEvaluation.model_validate_json(tool_call.function.arguments)

            logger.info(f"Content evaluated with overall score: {self.evaluation.overall_score}")

            return self.evaluation
        
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            raise

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
            
            logger.info("Optimizing content based on evaluation feedback...")

            # first generate the optimized content
            completion_content = client.chat.completions.create(
                model=model,
                messages=[
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
            )

            optimized_text = completion_content.choices[0].message.content
            
            # then get metadata about the optimized content
            completion_metadata = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": OPTIMIZATION_METADATA_PROMPT.format(
                            original_content=self.original_content,
                            optimized_content=optimized_text,
                            evaluation=self.evaluation.model_dump_json(indent=2)
                        )
                    }
                ],
                response_format={"type": "json_object"},
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "optimize_content",
                            "description": "provide metadata about the optimization",
                            "parameters": OptimizedContent.model_json_schema()
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "optimize_content"}}
            )

            tool_call = completion_metadata.choices[0].message.tool_calls[0]
            metadata = json.loads(tool_call.function.arguments)

            # Create the OptimizedContent object with the optimized text and metadata
            self.optimized_content = OptimizedContent(
                content=optimized_text,
                improvements_made=metadata["improvements_made"],
                expected_score_increase=metadata["expected_score_increase"]
            )

            logger.info(f"Content optimized with expected score increase: {self.optimized_content.expected_score_increase}")

            return self.optimized_content

        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            raise

    def run_workflow(self,
                    dataset_description: str,
                    data_quality: DataQuality,
                    analysis_goal: str,
                    target_audience: str,
                    required_length: int = 500,
                    focus_area: str = "actionable and insightful") -> OptimizedContent:
        """Run the complete evaluator-optimizer workflow."""
        start_time = time.time()
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

            end_time = time.time()
            logger.info(f"Workflow completed in {end_time - start_time:.2f} seconds")

            return {
                "original_content": original_content,
                "evaluation": evaluation,
                "optimized_content": optimized_content
            }

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            raise

# --------------------------------------------------------------
# Example usage
# --------------------------------------------------------------

if __name__ == "__main__":
    try:
        workflow = EvaluatorOptimizer()

        # Example scenario
        dataset_description = "Retail sales data for a clothing company over the past 3 years, including customer demographics, purchase history, and seasonal trends."
        data_quality = DataQuality.MEDIUM
        analysis_goal = "Identify opportunities to increase customer retention and lifetime value"
        target_audience = "Marketing team and senior management"
        
        result = workflow.run_workflow(
            dataset_description=dataset_description,
            data_quality=data_quality,
            analysis_goal=analysis_goal,
            target_audience=target_audience,
            required_length=800,
            focus_area="data-driven and actionable"
        )

        # Print the results
        print("\n\n=== EVALUATION RESULTS ===\n")
        print(f"Overall Score: {result['evaluation'].overall_score}/10")
        
        print("\nStrengths:")
        for strength in result['evaluation'].strengths:
            print(f"- {strength}")
        
        print("\nWeaknesses:")
        for weakness in result['evaluation'].weaknesses:
            print(f"- {weakness}")
        
        print("\n\n=== IMPROVEMENTS MADE ===\n")
        for improvement in result['optimized_content'].improvements_made:
            print(f"- {improvement}")
        
        print(f"\nExpected Score Increase: {result['optimized_content'].expected_score_increase} points")
        
        print("\n\n=== OPTIMIZED CONTENT ===\n")
        print(result['optimized_content'].content)
        
        # Save the results to files
        with open("original_content.txt", "w") as f:
            f.write(result['original_content'])
        
        with open("evaluation.txt", "w") as f:
            f.write(f"Overall Score: {result['evaluation'].overall_score}/10\n\n")
            f.write("Criteria Scores:\n")
            for criterion in result['evaluation'].criteria_scores:
                f.write(f"- {criterion.name}: {criterion.score}/10\n")
                f.write(f"  {criterion.feedback}\n\n")
            f.write("\nStrengths:\n")
            for strength in result['evaluation'].strengths:
                f.write(f"- {strength}\n")
            f.write("\nWeaknesses:\n")
            for weakness in result['evaluation'].weaknesses:
                f.write(f"- {weakness}\n")
            f.write("\nImprovement Suggestions:\n")
            for suggestion in result['evaluation'].improvement_suggestions:
                f.write(f"- {suggestion}\n")
        
        with open("optimized_content.txt", "w") as f:
            f.write(result['optimized_content'].content)
        
        logger.info("Results saved to files: original_content.txt, evaluation.txt, optimized_content.txt")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise