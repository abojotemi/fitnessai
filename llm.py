# llm.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

class LLMHandler:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
    
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
        sets, reps, rest periods, and form guidance."""

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
            ('system', "You are an expert text summarizer. Your job is to create a concise yet informative summary of workout plans, focusing on key exercises and important instructions."),
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

    def answer_question(self, query: str, info):

        messages = [
            ('system', "You are an gym assistant. You will be given a question. You must generate a detailed answer. You will also be given details about the person to let you answer better."),
            ('human', """
             - Name: {name}
             - Age: {age}
             - Sex: {sex}
             - Weight: {weight}
             - Height: {height}
             - Goals: {goals}
             - Country: {country}

             Message: {query}
             """),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm
        
        try:
            info_copy = info.model_dump().copy()
            info_copy.update({'query': query})
            response = chain.invoke(info_copy)
            return response.content
        except Exception as e:
            st.error(f"Error generating workout plan: {str(e)}")
            return None
    
     
    def analyze_diet(self, food_items, user_info):
        
        messages = ["""You are a helpful health assistant whose job is to strictly provide information about the food or fruits given by me.
        You can suggest healthier alternatives or things to add to make it healthier for people that fit my information?
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
        
        