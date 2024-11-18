from textwrap import dedent
from typing import Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate 
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_core.messages import HumanMessage
import streamlit as st
import logging
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt

from image_processing import process_image

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
        
        system_message = dedent("""You are a professional fitness trainer whose job is to create 
        personalized workout plans. You should provide detailed, safe, and effective workout 
        routines based on the user's profile and preferences. Include specific exercises, 
        sets, reps, rest periods, and form guidance. Make it a maximum of 300 words""")

        base_profile = dedent("""
        User Profile:
        - Name: {name}
        - Age: {age}
        - Sex: {sex}
        - Weight: {weight}
        - Height: {height}
        - Goals: {goals}
        - Country: {country}
        """)

        try:
            if workout_preferences:
                preference_template = dedent("""
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
                """)
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
        self.system_prompt = dedent("""You are an experienced fitness coach and personal trainer with extensive knowledge in:
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
        
        Remember the user's profile information to personalize your response. Make your answer short and straight to the point.""")

        self.human_template = dedent("""User Profile:
        - Name: {name}
        - Age: {age}
        - Sex: {sex}
        - Weight: {weight}kg
        - Height: {height}cm
        - Fitness Goals: {goals}
        - Country: {country}

        Question: {query}

        Please provide a detailed, personalized response considering the user's profile.""")

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
            PROMPT = dedent(f"""
            You are a virtual fitness coach and instructor. You will provide concise, informative, and engaging answers to fitness-related questions in a maximum of 10 seconds. Your tone should be friendly, professional, and motivating. Each response should be structured to deliver clear advice or information quickly. Keep the language simple and avoid jargon unless necessary, but use common fitness terms if they make the response more accurate.

            When crafting your response, consider the following guidelines:
            1. **Be Direct**: Answer the question directly without unnecessary introductions or fillers.
            2. **Be Encouraging**: Use a positive and motivating tone to engage the user and make them feel confident.
            3. **Stay Focused**: Limit the response to one or two key points. Avoid lengthy explanations or complex details.
            4. **Clarity**: Use clear and concise language, ensuring the advice is actionable and easy to understand.

            Example Responses:
            1. **Question**: "What is the best exercise for abs?"
            **Answer**: "The best exercise for your abs is the plank! It targets your core muscles effectively. Start with 30 seconds and gradually increase."
            2. **Question**: "How can I lose weight quickly?"
            **Answer**: "Focus on a calorie deficit and regular exercise, like cardio. Consistency is key! Track your food and stay active daily."
            3. **Question**: "How many times a week should I work out?"
            **Answer**: "Aim for at least 3-4 times a week, mixing strength training and cardio for balanced results."

            Remember, keep your responses concise and engaging for a short, 10-second video format. 
            AND MOST IMPORTANTLY GIVE YOUR ANSWER IN A PURE TEXT FORM. DO NOT MAKE IT MARKDOWN. 
            """
            )
            human_prompt = dedent(f"""User Profile:
                - Name: {user_info.name}
                - Age: {user_info.age}
                - Sex: {user_info.sex}
                - Weight: {user_info.weight}kg
                - Height: {user_info.height}cm
                - Fitness Goals: {user_info.goals}
                - Country: {user_info.country}
                
                QUESTION:
                
                {query}
                """)
            response = self.llm.invoke(PROMPT + human_prompt)
            
            if not response or not response.content:
                raise ValueError("Empty response from LLM")
            
            logger.info(f"Successfully processed question for user {context.name}")
            return response.content

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            st.error("Failed to process your question. Please try again later.")
            return None

        
    @staticmethod
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def video_analyzer_llm(title: str, query: str, context: str) -> Optional[str]:
        """Analyze video content with retry logic"""
        logger.info(f"Analyzing video content: {title}")
        
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-8b-latest",
                temperature=0.7,
            )
            
            messages = [
                ('system', dedent("""
                You are an intelligent video content analyzer. Your task is to provide accurate, relevant answers to questions about the video content using the provided context from the video transcript. Please follow these guidelines:
                
                1. Base your answers solely on the provided context
                2. If the context doesn't contain enough information to answer the question, clearly state that
                3. Include relevant quotes from the transcript when appropriate
                4. Maintain a natural, conversational tone while being informative
                
                Title: {title}
                Context: {context}
                
                Question: {question}
                
                Please provide a clear and concise answer based on the above context.""")),
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
        
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def analyze_diet(self, img, user_info):
        
            try:
                SYSTEM_PROMPT = dedent("""
                You are an intelligent AI assistant specialized in food recognition and health advice. Your task is to help users by analyzing images of food and providing useful information. Follow these steps when a user uploads an image of a food item:

                1. **Food Identification**: Analyze the uploaded image to identify the food item. Use visual recognition to determine the type of dish or ingredient shown in the image. If you are unsure or cannot identify it, ask the user for clarification.

                2. **Recipe Generation**: Once the food item is identified:
                - Provide a detailed recipe including ingredients, quantities, and step-by-step instructions.
                - Ensure the recipe is clear, easy to follow, and suitable for standard kitchen settings.

                3. **Health Effects**:
                - List the health effects of the main ingredients in the dish.
                - Mention any nutritional properties such as vitamins, minerals.

                4. **Healthy Alternatives**:
                - Suggest possible modifications or healthier alternatives to the dish. 
                - Focus on reducing unhealthy ingredients (e.g., sugar, refined flour, saturated fats).
                - Propose ingredient substitutions to make the dish more nutritious (e.g., using whole grains instead of refined, plant-based options, etc.).

                5. **Friendly and Helpful Tone**: Maintain a friendly and supportive tone. Be concise and informative in your responses.

                Example:
                - User uploads an image of a dish (e.g., pizza).
                - Identify the dish as pizza.
                - Provide a standard pizza recipe.
                - Explain health benefits of its main ingredients like tomatoes and cheese.
                - Suggest healthy alternatives like using whole-wheat crust or reducing cheese.

                Keep your answers comprehensive but concise.

                """)
                human_prompt = dedent(f"""User Profile:
                - Name: {user_info.name}
                - Age: {user_info.age}
                - Sex: {user_info.sex}
                - Weight: {user_info.weight}kg
                - Height: {user_info.height}cm
                - Fitness Goals: {user_info.goals}
                - Country: {user_info.country}
                Please provide a detailed, personalized response considering the user's profile.""")
                model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest") 
            # Process the image
                img_base64 = process_image(img)
                
                # Create the message with correct content format
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT + human_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    ]
                )
                
                # Get the response
                response = model.invoke([message])
                return response.content

            except Exception as e:
                logger.error(f"Error occurred: {str(e)}")
