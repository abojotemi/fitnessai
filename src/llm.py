from typing import Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate 
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import streamlit as st
import logging
from pydantic import BaseModel, Field
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up SQLite cache with error handling
try:
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))
    logger.info("Successfully initialized SQLite cache")
except Exception as e:
    logger.error(f"Failed to initialize SQLite cache: {str(e)}")

class QuestionContext(BaseModel):
    """Structured format for question context with validation"""
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=13, le=120)
    sex: str = Field(..., pattern="^(Male|Female|Other)$")
    weight: float = Field(..., ge=20.0, le=300.0)
    height: float = Field(..., ge=50.0, le=300.0)
    goals: str = Field(..., min_length=5)
    country: str
    query: str = Field(..., min_length=1)

class LLMHandler:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-8b-latest",
                temperature=0.7,
                retry_on_failure=True
            )
            self._setup_prompts()
            logger.info("Successfully initialized LLMHandler")
        except Exception as e:
            logger.error(f"Failed to initialize LLMHandler: {str(e)}")
            raise

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def generate_fitness_plan(self, user_profile: QuestionContext, 
                            workout_preferences: Optional[dict] = None) -> Optional[str]:
        """
        Generate personalized fitness plan with retry logic.
        
        Args:
            user_profile (QuestionContext): Validated user information
            workout_preferences (Optional[dict]): Optional workout preferences
        
        Returns:
            Optional[str]: Generated fitness plan or None if generation fails
        """
        logger.info(f"Generating fitness plan for user: {user_profile.name}")
        
        system_message = """You are a professional fitness trainer whose job is to create 
        personalized workout plans. You should provide detailed, safe, and effective workout 
        routines based on the user's profile and preferences. Include specific exercises, 
        sets, reps, rest periods, and form guidance. Make it a maximum of 300 words"""

        base_profile = """
        User Profile:
        - Name: {name}
        - Age: {age}
        - Sex: {sex}
        - Weight: {weight}
        - Height: {height}
        - Goals: {goals}
        - Country: {country}
        """

        try:
            if workout_preferences:
                preference_template = """
                Workout Preferences:
                - Available Equipment: {equipment}
                - Workout Duration: {duration} minutes
                - Weekly Frequency: {frequency}
                - Focus Areas: {focus_areas}
                - Limitations/Injuries: {limitations}
                
                Please provide a detailed workout plan including:
                1. Weekly schedule breakdown
                2. Detailed workouts with sets, reps, and rest periods
                3. Proper warm-up and cool-down routines
                4. Form guidance for each exercise
                5. Alternative exercises for each movement
                6. Progressive overload recommendations
                7. Safety tips and precautions
                """
                human_message = base_profile + preference_template
                prompt_data = {**user_profile.model_dump(), **workout_preferences}
            else:
                human_message = base_profile
                prompt_data = user_profile.model_dump()

            messages = [
                ("system", system_message),
                ("human", human_message),
            ]

            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | self.llm
            
            fitness_plan = chain.invoke(prompt_data)
            logger.info(f"Successfully generated fitness plan for user: {user_profile.name}")
            return fitness_plan

        except Exception as e:
            logger.error(f"Error generating fitness plan: {str(e)}", exc_info=True)
            st.error("Failed to generate fitness plan. Please try again later.")
            return None

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def summarizer(self, message: str) -> Optional[str]:
        """Summarize text content with retry logic."""
        logger.info("Starting text summarization")
        
        messages = [
            ('system', "You are an expert text summarizer. Your job is to create a concise yet informative summary of workout plans, maximum of 75 words, focusing on key exercises and important instructions."),
            ('human', "{msg}"),
        ]

        try:
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | self.llm
            response = chain.invoke({'msg': message})
            logger.info("Successfully summarized text")
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}", exc_info=True)
            st.error("Failed to summarize text. Please try again later.")
            return None

    def _setup_prompts(self) -> None:
        """Initialize prompt templates with enhanced context"""
        self.system_prompt = """You are an experienced fitness coach and personal trainer with extensive knowledge in:
        - Exercise physiology
        - Nutrition and diet planning
        - Strength training and conditioning
        - Injury prevention and rehabilitation
        - Mental wellness and motivation
        
        Guidelines for your responses:
        1. Provide scientifically backed information when possible
        2. Consider the user's specific context (age, goals, etc.)
        3. Include both immediate answers and long-term recommendations
        4. Highlight safety considerations when relevant
        5. Use clear, accessible language while maintaining expertise
        6. Break down complex concepts into digestible parts
        
        Remember the user's profile information to personalize your response. Make your answer short and straight to the point."""

        self.human_template = """User Profile:
        - Name: {name}
        - Age: {age}
        - Sex: {sex}
        - Weight: {weight}kg
        - Height: {height}cm
        - Fitness Goals: {goals}
        - Country: {country}

        Question: {query}

        Please provide a detailed, personalized response considering the user's profile."""

        self.messages = [
            ('system', self.system_prompt),
            ('human', self.human_template)
        ]
        
        self.prompt = ChatPromptTemplate.from_messages(self.messages)
        logger.info("Successfully set up prompt templates")

    def _validate_inputs(self, query: str, user_info: Any) -> Optional[QuestionContext]:
        """Validate inputs with enhanced error handling"""
        try:
            if not query or not query.strip():
                raise ValueError("Question cannot be empty")
            
            context = QuestionContext(
                name=user_info.name,
                age=user_info.age,
                sex=user_info.sex,
                weight=user_info.weight,
                height=user_info.height,
                goals=user_info.goals,
                country=user_info.country,
                query=query.strip()
            )
            logger.info("Successfully validated user inputs")
            return context
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}", exc_info=True)
            return None

    def _format_response(self, raw_response: str) -> str:
        """Format the LLM response with error handling"""
        try:
            sections = {
                "Answer": "",
                "Additional Recommendations": "",
                "Safety Notes": "",
            }
            
            current_section = "Answer"
            for line in raw_response.split('\n'):
                if any(section in line for section in sections.keys()):
                    current_section = line.strip(':')
                else:
                    sections[current_section] += line + '\n'

            formatted_response = f"""
                ### Direct Answer
                {sections['Answer']}

                ### Additional Recommendations
                {sections['Additional Recommendations']}

                ### Safety Notes
                {sections['Safety Notes']}
                """
            logger.info("Successfully formatted response")
            return formatted_response
        except Exception as e:
            logger.error(f"Response formatting error: {str(e)}", exc_info=True)
            return raw_response

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def answer_question(self, query: str, user_info: Any) -> Optional[str]:
        """Generate answer to fitness question with retry logic"""
        logger.info(f"Processing question for user: {user_info.name}")
        
        try:
            context = self._validate_inputs(query, user_info)
            if not context:
                st.error("Invalid input data. Please check your question and profile information.")
                return None

            chain = self.prompt | self.llm
            response = chain.invoke(context.model_dump())
            
            if not response or not response.content:
                raise ValueError("Empty response from LLM")
            
            formatted_response = self._format_response(response.content)
            logger.info(f"Successfully processed question for user {context.name}")
            return formatted_response

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            st.error("Failed to process your question. Please try again later.")
            return None

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def get_follow_up_questions(self, original_query: str, response: str) -> List[str]:
        """Generate follow-up questions with retry logic"""
        logger.info("Generating follow-up questions")
        
        try:
            follow_up_prompt = f"""
            Based on the original question: "{original_query}"
            And the response provided: "{response}"
            
            Generate 3 relevant follow-up questions that the user might want to ask.
            Return them in a simple list format.
            """
            
            chain = ChatPromptTemplate.from_messages([
                ("system", "Generate relevant follow-up questions for fitness-related queries."),
                ("human", follow_up_prompt)
            ]) | self.llm
            
            result = chain.invoke({})
            questions = [q.strip() for q in result.content.split('\n') if q.strip()]
            logger.info("Successfully generated follow-up questions")
            return questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}", exc_info=True)
            return []
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def analyze_diet(self, food_items: dict, user_info: QuestionContext) -> Optional[str]:
        """Analyze diet with retry logic"""
        logger.info(f"Analyzing diet for user: {user_info.name}")
        
        messages = ["""You are a helpful health assistant whose job is to strictly provide information about the food or fruits given by me.
        You can suggest healthier alternatives or things to add to make it healthier for people that fit my information? You are to answer with a maximum of 250 words.
        
        Food information: {food_items}
        User Information:
        - Name: {name}
        - Age: {age}
        - Sex: {sex}
        - Weight: {weight}
        - Height: {height}
        - Goals: {goals}
        - Country: {country}
        """]
        
        try:
            prompt = ChatPromptTemplate.from_messages(messages)
            info_copy = user_info.model_dump().copy()
            info_copy.update({'food_items': food_items['label']})
            
            chain = prompt | self.llm
            response = chain.invoke(info_copy)
            logger.info(f"Successfully analyzed diet for user: {user_info.name}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing diet: {str(e)}", exc_info=True)
            st.error("Failed to analyze diet. Please try again later.")
            return None
        
    @staticmethod
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def video_analyzer_llm(title: str, query: str, context: str) -> Optional[str]:
        """Analyze video content with retry logic"""
        logger.info(f"Analyzing video content: {title}")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
            )
            
            messages = [
                ('system', """
                You are an intelligent video content analyzer. Your task is to provide accurate, relevant answers to questions about the video content using the provided context from the video transcript. Please follow these guidelines:
                
                1. Base your answers solely on the provided context
                2. If the context doesn't contain enough information to answer the question, clearly state that
                3. Include relevant quotes from the transcript when appropriate
                4. Maintain a natural, conversational tone while being informative
                
                Title: {title}
                Context: {context}
                
                Question: {question}
                
                Please provide a clear and concise answer based on the above context."""),
                ('human', f"Context: {context}\n\nQuestion: {query}")
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | llm
            
            response = chain.invoke({
                'title': title,
                'context': context,
                'question': query
            })
            
            logger.info(f"Successfully analyzed video content: {title}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}", exc_info=True)
            return None