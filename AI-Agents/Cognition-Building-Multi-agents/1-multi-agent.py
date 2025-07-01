"""
The system creates a research report generation workflow:

1. Research Agent: Gathers information on a topic
2. Analysis Agent: Analyzes the research data
3. Writer Agent: Creates a comprehensive report
4. Reviewer Agent: Reviews and validates the final output
"""

# ============================================
# Import necessary libraries and configure environment
# ==============================================

import os
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import json
from abc import ABC, abstractmethod
import time
import asyncio

# ============================================
# global constants and configurations
# ============================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOPIC = "Impact of Artificial Intelligence on Future Job Markets"

# ============================================
# core enums and data structures
# ============================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"

class AgentType(Enum):
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"

@dataclass
class TaskState:
    task_id: str
    status: TaskStatus
    agent_id: str
    created_at: datetime
    updated_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

# ============================================
# Pydantic models for api interactions  
# ============================================

class ResearchResult(BaseModel):
    topic: str = Field(description="The topic of research")
    key_findings: List[str] = Field(description="list of key findings from the research")
    sources: List[str] = Field(description="list of sources used in the research")
    confidence_score: float = Field(description="Confidence score of the research quality, between 0 and 1")
    recommendations: List[str] = Field(description="list of recommendations based on the research findings")

class AnalysisResult(BaseModel):
    data_summary: str = Field(description="Summary of the analyzed data")
    insights: List[str] = Field(description="list the key insights derived from the analysis")
    opportunities: List[str] = Field(description="list of opportunities identified from the analysis")
    risk_factors: List[str] = Field(description="list of risk factors identified during the analysis")
    trends: List[str] = Field(description="list of trends observed in the data")

class ContentResult(BaseModel):
    title: str = Field(description="Title of the content")
    content: str = Field(description="The main content of the report")
    key_points: List[str] = Field(description="list of key points highlighted in the content")
    word_count: int = Field(description="Total word count of the content")
    target_audience: str = Field(description="Target audience for the content")

class ReviewResult(BaseModel):
    overall_score: float = Field(description="Overall quality score of the content, between 0 and 10")
    strengths: List[str] = Field(description="list of strengths identified in the content")
    weaknesses: List[str] = Field(description="list of weaknesses identified in the content")
    suggestions: List[str] = Field(description="list of suggestions for improvement")
    approved: bool = Field(description="Whether the content is approved")

# ============================================
# state management system
# ============================================

class StateManager:
    """state management system to track task states and provide persistence."""

    def __init__(self, persistence_file: str = None):
        if persistence_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            persistence_file = os.path.join(current_dir, 'task_states.json')

        self.states: Dict[str, TaskState] = {}
        self.persistence_file = persistence_file
        self.load_states()

    def create_task(self, task_id: str, agent_id: str, dependencies: List[str] = None):
        """Create a new task with the given ID and agent."""
        
        task_state = TaskState(
            task_id=task_id,
            agent_id=agent_id,
            dependencies=dependencies if dependencies else [],
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.states[task_id] = task_state
        self.save_states()

        return task_state

    def update_task(self, task_id: str, **updates) -> Optional[TaskState]:
        """Update the task state with the provided updates."""
        
        if task_id not in self.states:
            return None
        
        task_state = self.states[task_id]
        for key, value in updates.items():
            if hasattr(task_state, key):
                setattr(task_state, key, value)

        task_state.updated_at = datetime.now()
        self.save_states()

        return task_state

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Retrieve the task state by task ID."""

        return self.states.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskState]:
        """Retrieve all tasks with the specified status."""
        
        return [task for task in self.states.values() if task.status == status]

    def are_dependencies_completed(self, task_id: str) -> bool:
        """Check if all dependencies of a task are completed."""
        
        task = self.get_task(task_id)
        if not task or not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
            
        return True

    def get_ready_tasks(self) -> List[TaskState]:
        """Retrieve tasks that are ready to be processed (i.e., dependencies are completed)."""

        ready_tasks = []
        for task in self.get_tasks_by_status(TaskStatus.PENDING):
            if self.are_dependencies_completed(task.task_id):
                ready_tasks.append(task)

        return ready_tasks

    def save_states(self):
        """Save the current states to a JSON file."""
        
        try:
            data = {}
            for task_id, task_state in self.states.items():
                data[task_id] = {
                    **asdict(task_state),
                    "created_at": task_state.created_at.isoformat(),
                    "updated_at": task_state.updated_at.isoformat(),
                    "status": task_state.status.value
                }

            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save states: {e}")

    def load_states(self):
        """load the states from a JSON file."""

        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)

                for task_id, task_data in data.items():
                    task_state = TaskState(
                        task_id=task_id,
                        agent_id=task_data['agent_id'],
                        status=TaskStatus(task_data['status']),
                        dependencies=task_data.get('dependencies', []),
                        created_at=datetime.fromisoformat(task_data['created_at']),
                        updated_at=datetime.fromisoformat(task_data['updated_at']),
                        retry_count=task_data.get('retry_count', 0),
                        max_retries=task_data.get('max_retries', 3),
                        result=task_data.get('result'),
                        error=task_data.get('error'),
                        metadata=task_data.get('metadata', {})
                    )
                    self.states[task_id] = task_state
        except Exception as e:
            logger.error(f"Failed to load states: {e}")

# ============================================
# Agent base class
# ============================================

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, agent_id: str, agent_type: AgentType, state_manager: StateManager):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state_manager = state_manager
        self.logger = logging.getLogger(f"{self.__class__.__name__}({agent_id})")

    @abstractmethod
    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Execute the task assigned to this agent."""
        pass
    
    async def handle_task(self, task_id: str, **kwargs) -> bool:
        """handle the task execution and update the state."""

        try:
            self.logger.info(f"Handling task_id {task_id} for agent {self.agent_id}")

            # update task state to in progress
            self.state_manager.update_task(
                task_id=task_id,
                status=TaskStatus.IN_PROGRESS,
                updated_at=datetime.now()
            )

            # execute the task
            result = await self.execute_task(task_id, **kwargs)

            # update task state to completed
            self.state_manager.update_task(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                updated_at=datetime.now()
            )
            self.logger.info(f"Task {task_id} completed successfully.")

            return True
        
        except Exception as e:
            self.logger.error(f"Error handling task {task_id}: {e}")
            
            # get current task state
            task_state = self.state_manager.get_task(task_id)

            if task_state and task_state.retry_count < task_state.max_retries:
                # mark task as retry needed
                self.state_manager.update_task(
                    task_id=task_id,
                    status=TaskStatus.RETRY_NEEDED,
                    retry_count=task_state.retry_count + 1,
                    error=str(e)
                )
            else: 
                # mark task as failed
                self.state_manager.update_task(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
                self.logger.error(f"Task {task_id} failed after {task_state.retry_count} retries.")

            return False

    async def make_llm_call(self, prompt: str, response_model: BaseModel = None, **kwargs) -> Any:
        """make a call to the LLM with the given prompt and return the response."""

        try:
            if response_model:
                prompt += "\n\nPlease provide your response in JSON format."

            messages = [{"role": "user", "content": prompt}]

            if response_model:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "structured_response",
                            "description": f"Generate a structured response based on the provided model: {response_model.__name__}",
                            "parameters": response_model.model_json_schema()
                        }
                    }
                ]

                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "structured_response"}},
                    response_format = {"type": "json_object"},
                    **kwargs
                )

                tool_call = completion.choices[0].message.tool_calls[0]

                return response_model.model_validate_json(tool_call.function.arguments)
            else:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    **kwargs
                )

                return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise e
        

# ============================================
# prompt templates
# ============================================

RESEARCH_PROMPT_TEMPLATE = """
Conduct {research_depth} research on the following topic: {topic}
        
Provide:
1. Key findings with supporting evidence
2. Relevant sources and references
3. Current trends and developments
4. Expert opinions and insights
5. Recommendations for further investigation

Ensure the research is thorough, accurate, and up-to-date.
"""

ANALYSIS_PROMPT_TEMPLATE = """
Perform {analysis_type} analysis on the following data:
        
Data Source: {data_source}
{research_data}

Provide:
1. Data summary and overview
2. Identified trends and patterns
3. Key insights and implications
4. Risk factors and concerns
5. Opportunities for improvement or growth

Use analytical thinking and data-driven approaches.
"""

WRITER_PROMPT_TEMPLATE = """
Create a {length} {content_type} for {target_audience} audience based on the following data:
        
{research_data}
{analysis_data}

Requirements:
1. Clear and engaging title
2. Well-structured content with logical flow
3. Key points highlighted
4. Appropriate tone for the target audience
5. Actionable insights and recommendations

The content should be informative, well-written, and professionally formatted.
"""

REVIEWER_PROMPT_TEMPLATE = """
Review the following content based on these criteria: {', '.join(review_criteria)}

Content to Review:
{content_data}

Provide:
1. Overall quality score (0-10)
2. Specific strengths identified
3. Weaknesses or areas for improvement
4. Actionable suggestions for enhancement
5. Final approval recommendation

Be thorough, constructive, and objective in your review.
"""

# ============================================
# specific agent implementations
# ============================================

class ResearchAgent(BaseAgent):
    """Agent responsible for gathering information on a specific topic."""

    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        topic = kwargs.get("topic", "")
        research_depth = kwargs.get("research_depth", "comprehensive")

        prompt = RESEARCH_PROMPT_TEMPLATE.format(
            topic=topic,
            research_depth=research_depth
        )

        result = await self.make_llm_call(
            prompt=prompt,
            response_model=ResearchResult
        )

        return result.model_dump()        

class AnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing research data."""

    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        data_source = kwargs.get("data_source", "")
        analysis_type = kwargs.get("analysis_type", "comprehensive")
        
        # get research data from state manager
        research_data = ""
        task_state = self.state_manager.get_task(task_id)
        if task_state and task_state.dependencies:
            for dep_id in task_state.dependencies:
                dep_task = self.state_manager.get_task(dep_id)
                if dep_task and dep_task.result:
                    research_data += f"\nResearch Data from {dep_id}:\n{json.dumps(dep_task.result, indent=2)}\n"
        
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            data_source=data_source,
            analysis_type=analysis_type,
            research_data=research_data
        )

        result = await self.make_llm_call(
            prompt=prompt,
            response_model=AnalysisResult  
        )

        return result.model_dump()

class WriterAgent(BaseAgent):
    """agent responsible for creating a comprehensive report based on research and analysis data."""

    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        content_type = kwargs.get("content_type", "report")
        target_audience = kwargs.get("target_audience", "general")
        length = kwargs.get("length", "meduim")

        # get research and analysis data from state manager
        research_data = ""
        analysis_data = ""
        task_state = self.state_manager.get_task(task_id)
        if task_state and task_state.dependencies:
            for dep_id in task_state.dependencies:
                dep_task = self.state_manager.get_task(dep_id)
                if dep_task and dep_task.result:
                    # check agent type based on agent_id naming convention
                    agent_id = dep_task.agent_id
                    if 'research' in agent_id.lower():
                        research_data += f"\nResearch Data from {dep_id}:\n{json.dumps(dep_task.result, indent=2)}\n"
                    elif 'analy' in agent_id.lower():
                        analysis_data += f"\nAnalysis Data from {dep_id}:\n{json.dumps(dep_task.result, indent=2)}\n"

        prompt = WRITER_PROMPT_TEMPLATE.format(
            content_type=content_type,
            target_audience=target_audience,
            length=length,
            research_data=research_data,
            analysis_data=analysis_data
        )

        result = await self.make_llm_call(
            prompt=prompt,
            response_model=ContentResult
        )

        return result.model_dump()

class ReviewerAgent(BaseAgent):

    """Agent responsible for reviewing and validating the final report."""

    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        review_criteria = kwargs.get("review_criteria", ["clarity", "accuracy", "engagement"])
        
        # get content data from state manager
        content_data = ""
        task_state = self.state_manager.get_task(task_id)
        if task_state and task_state.dependencies:
            for dep_id in task_state.dependencies:
                dep_task = self.state_manager.get_task(dep_id)
                if dep_task and dep_task.result:
                    content_data += f"\nContent Data from {dep_id}:\n{json.dumps(dep_task.result, indent=2)}\n"

        prompt = REVIEWER_PROMPT_TEMPLATE.format(
            review_criteria=review_criteria,
            content_data=content_data
        )

        result = await self.make_llm_call(
            prompt=prompt,
            response_model=ReviewResult
        )

        return result.model_dump()
    
# ============================================
# Coordinator Agent
# ============================================

class CoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating the workflow and managing task dependencies."""

    def __init__(self, agent_id: str, state_manager: StateManager):
        super().__init__(agent_id, AgentType.COORDINATOR, state_manager)
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """Register an agent to the coordinator."""
        
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} of type {agent.agent_type.value}")

    async def execute_workflow_tasks(self, task_order: List[str], task_configs: Dict[str, Dict[str, Any]]):
        """"Execute the tasks in the specified order, respecting dependencies."""

        for task_id in task_order:
            # wait for dependencies to be completed
            while not self.state_manager.are_dependencies_completed(task_id):
                await asyncio.sleep(1)

            # execute the task
            config = task_configs[task_id]
            agent = self.agents.get(config['agent'])

            if agent:
                self.logger.info(f"Executing task {task_id} with agent {agent.agent_id}")
                await agent.handle_task(task_id=task_id, **config['kwargs'])
            else:
                self.logger.error(f"No agent found for task {task_id} with agent {config['agent']}")
                self.state_manager.update_task(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"No agent found for {config['agent']}",
                    updated_at=datetime.now()
                )

    async def execute_research_analysis_report(self, topic: str) -> Dict[str, Any]:
        """execute the research analysis report workflow. (research -> analysis -> writing -> review)"""

        workflow_id = f"workflow_{int(time.time())}"

        # create tasks with dependencies
        research_task_id = f"{workflow_id}_research"
        analysis_task_id = f"{workflow_id}_analysis"
        writing_task_id = f"{workflow_id}_writing"
        review_task_id = f"{workflow_id}_review"

        # create tasks
        self.state_manager.create_task(task_id=research_task_id, agent_id="research_agent", dependencies=[])
        self.state_manager.create_task(task_id=analysis_task_id, agent_id="analyzer_agent", dependencies=[research_task_id])
        self.state_manager.create_task(task_id=writing_task_id, agent_id="writer_agent", dependencies=[research_task_id, analysis_task_id])
        self.state_manager.create_task(task_id=review_task_id, agent_id="reviewer_agent", dependencies=[writing_task_id])   

        # execute workflow
        task_configs = {
            research_task_id: {
                'agent': 'research_agent',
                'kwargs': {'topic': topic, 'research_depth': 'comprehensive'}
            },
            analysis_task_id: {
                'agent': 'analyzer_agent',
                'kwargs': {'data_source': f'research on {topic}', 'analysis_type': 'comprehensive'}
            },
            writing_task_id: {
                'agent': 'writer_agent',
                'kwargs': {
                    'content_type': 'report',
                    'target_audience': 'business stakeholders',
                    'length': 'comprehensive',
                }
            },
            review_task_id: {
                'agent': 'reviewer_agent',
                'kwargs': {
                    'criteria': ['clarity', 'accuracy', 'engagement', 'completeness', 'actionability'],
                }
            }
        }

        # execute tasks in dependencies order
        await self.execute_workflow_tasks([research_task_id, analysis_task_id, writing_task_id, review_task_id], task_configs)

        # collect results
        results = {}
        for task_id in [research_task_id, analysis_task_id, writing_task_id, review_task_id]:
            task_state = self.state_manager.get_task(task_id)
            if task_state and task_state.result:
                results[task_id] = task_state.result

        return {
            "workflow_id": workflow_id,
            "topic": topic,
            "status": "completed",
            "results": results
        }

    async def execute_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Execute the coordination task by managing the workflow."""
        
        workflow_type = kwargs.get("workflow_type", "research_analysis_report")
        topic = kwargs.get("topic", "")

        if workflow_type == "research_analysis_report":
            return await self.execute_research_analysis_report(topic=topic)
        else:
            self.logger.error(f"Unknown workflow type: {workflow_type}")
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
    async def monitor_and_recovery(self):
        """Monitor tasks and recover from failures."""
        
        while True:
            retry_tasks = self.state_manager.get_tasks_by_status(TaskStatus.RETRY_NEEDED)
            for task in retry_tasks:
                self.logger.info(f"Retrying task {task.task_id} for agent {task.agent_id}")

                # reset task state to pending
                self.state_manager.update_task(
                    task_id=task.task_id,
                    status=TaskStatus.PENDING,
                    updated_at=datetime.now()
                )

                # find the agent and handle the task
                agent = self.agents.get(task.agent_id)
                if agent:
                    await asyncio.sleep(2 ** task.retry_count)  
                    await agent.handle_task(task_id=task.task_id)
                else:
                    self.logger.error(f"No agent found for task {task.task_id} with agent {task.agent_id}")

            await asyncio.sleep(5)

# ============================================
# multi-agent system manager
# ============================================

class MultiAgentSystem:
    """Manager for the multi-agent system, coordinating agents and tasks."""

    def __init__(self):
        self.state_manager = StateManager()
        self.coordinator = CoordinatorAgent(agent_id='coordinator', state_manager=self.state_manager)

        # initialize specialized agents
        self.research_agent = ResearchAgent(agent_id='research_agent', agent_type=AgentType.RESEARCHER, state_manager=self.state_manager)
        self.analyzer_agent = AnalyzerAgent(agent_id='analyzer_agent', agent_type=AgentType.ANALYZER, state_manager=self.state_manager)
        self.writer_agent = WriterAgent(agent_id='writer_agent', agent_type=AgentType.WRITER, state_manager=self.state_manager)
        self.reviewer_agent = ReviewerAgent(agent_id='reviewer_agent', agent_type=AgentType.REVIEWER, state_manager=self.state_manager)

        # register agents with the coordinator
        self.coordinator.register_agent(self.research_agent)
        self.coordinator.register_agent(self.analyzer_agent)
        self.coordinator.register_agent(self.writer_agent)
        self.coordinator.register_agent(self.reviewer_agent)

        self.logger = logging.getLogger("MultiAgentSystem")

    async def execute_research_workflow(self, topic: str) -> Dict[str, Any]:
        """Execute the research workflow with the given topic."""
        
        self.logger.info(f"Starting research workflow for topic: {topic}")
        
        # create a task for the coordinator
        coordinator_task_id = f"coordinator_task_{int(time.time())}"
        self.state_manager.create_task(task_id=coordinator_task_id, agent_id='coordinator', dependencies=[])

        # execute the coordinator task
        result = await self.coordinator.handle_task(
            task_id=coordinator_task_id,
            workflow_type="research_analysis_report",
            topic=topic
        )

        if result:
            self.logger.info(f"Research workflow completed successfully for topic: {topic}")
            task_state = self.state_manager.get_task(coordinator_task_id)

            return task_state.result if task_state else {}
        else:
            self.logger.error(f"Research workflow failed for topic: {topic}")
            return {"error": "Workflow execution failed."}
        
    async def start_monitoring(self) -> Dict[str, Any]:
        """Start monitoring and recovery for the system."""
        
        self.logger.info("Starting system monitoring and recovery...")
        await self.coordinator.monitor_and_recovery()
        return {"status": "monitoring_started"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the system."""
        
        status = {
            'total_tasks': len(self.state_manager.states),
            'by_status': {},
            'recent_tasks': []
        }

        # count tasks by status
        for status_type in TaskStatus:
            count = len(self.state_manager.get_tasks_by_status(status_type))
            status['by_status'][status_type.value] = count

        # get recent tasks
        recent_tasks = sorted(
            self.state_manager.states.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )[:10]

        status['recent_tasks'] = [
            {
                'task_id': task.task_id,
                'status': task.status.value,
                'agent_id': task.agent_id,
                'updated_at': task.updated_at.isoformat(),
                
            } 
            for task in recent_tasks
        ]
        return status
    

# ============================================
# saving the results to a file
# ============================================

def format_research_results(f, results: Dict[str, Any]):
    """Format research results section."""
    f.write("RESEARCH FINDINGS\n")
    f.write("-" * 40 + "\n\n")
    
    f.write(f"Topic: {results.get('topic', 'N/A')}\n")
    f.write(f"Confidence Score: {results.get('confidence_score', 'N/A')}\n\n")
    
    f.write("Key Findings:\n")
    for i, finding in enumerate(results.get('key_findings', []), 1):
        f.write(f"  {i}. {finding}\n")
    f.write("\n")
    
    f.write("Sources:\n")
    for i, source in enumerate(results.get('sources', []), 1):
        f.write(f"  {i}. {source}\n")
    f.write("\n")
    
    f.write("Recommendations:\n")
    for i, rec in enumerate(results.get('recommendations', []), 1):
        f.write(f"  {i}. {rec}\n")

def format_analysis_results(f, results: Dict[str, Any]):
    """Format analysis results section."""
    f.write("ANALYSIS RESULTS\n")
    f.write("-" * 40 + "\n\n")
    
    f.write("Data Summary:\n")
    f.write(f"{results.get('data_summary', 'N/A')}\n\n")
    
    f.write("Key Insights:\n")
    for i, insight in enumerate(results.get('insights', []), 1):
        f.write(f"  {i}. {insight}\n")
    f.write("\n")
    
    f.write("Identified Trends:\n")
    for i, trend in enumerate(results.get('trends', []), 1):
        f.write(f"  {i}. {trend}\n")
    f.write("\n")
    
    f.write("Opportunities:\n")
    for i, opp in enumerate(results.get('opportunities', []), 1):
        f.write(f"  {i}. {opp}\n")
    f.write("\n")
    
    f.write("Risk Factors:\n")
    for i, risk in enumerate(results.get('risk_factors', []), 1):
        f.write(f"  {i}. {risk}\n")

def format_writing_results(f, results: Dict[str, Any]):
    """Format writing results section."""
    f.write("WRITTEN CONTENT\n")
    f.write("-" * 40 + "\n\n")
    
    f.write(f"Title: {results.get('title', 'N/A')}\n")
    f.write(f"Target Audience: {results.get('target_audience', 'N/A')}\n")
    f.write(f"Word Count: {results.get('word_count', 'N/A')}\n\n")
    
    f.write("Content:\n")
    f.write("-" * 20 + "\n")
    f.write(f"{results.get('content', 'N/A')}\n\n")
    
    f.write("Key Points:\n")
    for i, point in enumerate(results.get('key_points', []), 1):
        f.write(f"  {i}. {point}\n")

def format_review_results(f, results: Dict[str, Any]):
    """Format review results section."""
    f.write("QUALITY REVIEW\n")
    f.write("-" * 40 + "\n\n")
    
    f.write(f"Overall Score: {results.get('overall_score', 'N/A')}/10\n")
    f.write(f"Approved: {'✅ Yes' if results.get('approved', False) else '❌ No'}\n\n")
    
    f.write("Strengths:\n")
    for i, strength in enumerate(results.get('strengths', []), 1):
        f.write(f"  ✓ {strength}\n")
    f.write("\n")
    
    f.write("Weaknesses:\n")
    for i, weakness in enumerate(results.get('weaknesses', []), 1):
        f.write(f"  ⚠ {weakness}\n")
    f.write("\n")
    
    f.write("Suggestions for Improvement:\n")
    for i, suggestion in enumerate(results.get('suggestions', []), 1):
        f.write(f"  💡 {suggestion}\n")
        
def save_results_to_file(results: Dict[str, Any], filename: str = None):
    """Save the results to a JSON file."""
    
    if filename is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, 'research_results.txt')
    
    try:
        with open(filename, 'w') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("MULTI-AGENT RESEARCH WORKFLOW RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Workflow metadata
            f.write("WORKFLOW INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Workflow ID: {results.get('workflow_id', 'N/A')}\n")
            f.write(f"Topic: {results.get('topic', 'N/A')}\n")
            f.write(f"Status: {results.get('status', 'N/A')}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Results from each agent
            workflow_results = results.get('results', {})
            
            for task_id, task_result in workflow_results.items():
                # Determine the phase based on task_id
                if 'research' in task_id:
                    phase = "RESEARCH PHASE"
                    icon = "🔍"
                elif 'analysis' in task_id:
                    phase = "ANALYSIS PHASE"
                    icon = "📊"
                elif 'writing' in task_id:
                    phase = "WRITING PHASE"
                    icon = "✍️"
                elif 'review' in task_id:
                    phase = "REVIEW PHASE"
                    icon = "✅"
                else:
                    phase = "UNKNOWN PHASE"
                    icon = "❓"
                
                f.write(f"{icon} {phase}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Task ID: {task_id}\n\n")
                
                # Format the specific content based on phase
                if 'research' in task_id:
                    format_research_results(f, task_result)
                elif 'analysis' in task_id:
                    format_analysis_results(f, task_result)
                elif 'writing' in task_id:
                    format_writing_results(f, task_result)
                elif 'review' in task_id:
                    format_review_results(f, task_result)
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Footer
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Results saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save results to file {filename}: {e}")
        return None

    

# ============================================
# Example usage
# ============================================

async def main():
    """Main function to demonstrate the multi-agent system workflow."""

    # initialize the multi-agent system
    system = MultiAgentSystem()

    # execute a research workflow
    topic = TOPIC

    try:
        # start monitoring in the background
        monitoring_task = asyncio.create_task(system.start_monitoring())

        # execute the research workflow
        result = await system.execute_research_workflow(topic=topic)

        print("\n=== Workflow Results ===")
        print(json.dumps(result, indent=2, default=str))
        
        # Save results to file
        print("\n=== Saving Results to File ===")
        filename = save_results_to_file(result)
        if filename:
            print(f"✅ Results saved to: {filename}")
        else:
            print("❌ Failed to save results to file")
        
        # Display system status
        print("\n=== System Status ===")
        status = system.get_system_status()
        print(json.dumps(status, indent=2))
        
        # Cancel monitoring
        monitoring_task.cancel()

    except KeyboardInterrupt:
        print("Workflow execution interrupted by user.")
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        logger.exception("Error during workflow execution")

if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    asyncio.run(main())
    