from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import streamlit as st
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
class QuestionContext(BaseModel):
    """Structured format for question context"""
    name: str
    age: int
    sex: str
    weight: float
    height: float
    goals: str
    country: str
    query: str
    
class LLMHandler:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
        self._setup_prompts()

    
    def generate_fitness_plan(self, user_profile, workout_preferences=None) -> str:
        """
        Generate personalized fitness plan.
        
        Args:
            user_profile: UserInfo object containing basic user information
            workout_preferences: Optional dict containing workout-specific preferences
        
        Returns:
            str: Generated fitness plan
        """
        # Base system message
        system_message = """You are a professional fitness trainer whose job is to create 
        personalized workout plans. You should provide detailed, safe, and effective workout 
        routines based on the user's profile and preferences. Include specific exercises, 
        sets, reps, rest periods, and form guidance. Make it a maximum of 300 words"""

        # Base user profile template
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

        # Add workout preferences if provided
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
            # Combine user profile with preferences
            human_message = base_profile + preference_template
            # Merge profile and preferences data
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
        
        try:
            fitness_plan = chain.invoke(prompt_data)
            return fitness_plan
        except Exception as e:
            logger.error(f"Error generating fitness plan: {str(e)}")
            st.error(f"Error generating fitness plan: {str(e)}")
            return None

    def summarizer(self, _message: str) -> str:
        """Summarize text content."""
        messages = [
            ('system', "You are an expert text summarizer. Your job is to create a concise yet informative summary of workout plans, maximum of 75 words, focusing on key exercises and important instructions."),
            ('human', "{msg}"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({'msg': _message})
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            st.error(f"Error summarizing text: {str(e)}")
            return None

    def _setup_prompts(self):
        """Initialize prompt templates"""
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

    def _validate_inputs(self, query: str, user_info: Any) -> QuestionContext | None:
        """Validate inputs and create structured context"""
        try:
            if not query or not query.strip():
                raise ValueError("Question cannot be empty")
            
            # Convert user info to structured format
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
            return context
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return None

    def _format_response(self, raw_response: str) -> str:
        """Format the LLM response for better readability"""
        try:
            # Add sections if they don't exist
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

            # Format the response in Markdown
            formatted_response = f"""
                ### Direct Answer
                {sections['Answer']}

                ### Additional Recommendations
                {sections['Additional Recommendations']}

                ### Safety Notes
                {sections['Safety Notes']}
                """
            return formatted_response
        except Exception as e:
            logger.error(f"Response formatting error: {str(e)}")
            return raw_response

    def answer_question(self, query: str, user_info: Any) -> str | None:
        """
        Generate a detailed answer to a fitness-related question considering user context.
        
        Args:
            query (str): The user's question
            user_info (Any): User profile information
            
        Returns:
            Optional[str]: Formatted response or None if error occurs
        """
        try:
            # Validate inputs
            context = self._validate_inputs(query, user_info)
            if not context:
                st.error("Invalid input data. Please check your question and profile information.")
                return None

            # Create chain and generate response
            chain = self.prompt | self.llm
            
            # Add rate limiting if needed
            # time.sleep(1)  # Basic rate limiting
            
            response = chain.invoke(context.model_dump())
            
            if not response or not response.content:
                raise ValueError("Empty response from LLM")
            
            # Format and return response
            formatted_response = self._format_response(response.content)
            
            # Log successful query
            logger.info(f"Successfully processed question for user {context.name}")
            
            return formatted_response

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None

    def get_follow_up_questions(self, original_query: str, response: str) -> list[str]:
        """Generate relevant follow-up questions based on the original query and response"""
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
            
            # Parse the response into a list of questions
            questions = [q.strip() for q in result.content.split('\n') if q.strip()]
            return questions[:3]  # Return top 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
     
    def analyze_diet(self, food_items, user_info):
        
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
        prompt = ChatPromptTemplate.from_messages(messages)
        try:
            info_copy = user_info.model_dump().copy()
            info_copy.update({'food_items': food_items['label']})
            chain = prompt | self.llm
            response = chain.invoke(info_copy)
            return response.content
        except Exception as e:
            st.error(f"Error analyzing diet: {str(e)}")
            print(f'Error at llm: {str(e)}')
            return None
        
        