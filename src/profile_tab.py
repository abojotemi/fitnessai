import pycountry
from pydantic import ValidationError
import streamlit as st
import logging
from analytics_tab import log_user_interaction
from config import UserInfo
from typing import List, Optional

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Profile:
    def __init__(self):
        self._cached_countries: Optional[List[str]] = None

    def _get_countries(self) -> List[str]:
        """Cache and return list of countries"""
        if self._cached_countries is None:
            try:
                self._cached_countries = sorted([country.name for country in pycountry.countries])
                logger.info("Successfully loaded country list")
            except Exception as e:
                logger.error(f"Error loading country list: {str(e)}")
                self._cached_countries = ["United States"]  # Fallback
        return self._cached_countries

    def _get_default_country_index(self, countries: List[str]) -> int:
        """Get index of default country with error handling"""
        try:
            if st.session_state.user_info and st.session_state.user_info.country:
                return countries.index(st.session_state.user_info.country)
            return countries.index("United States")  # Default fallback
        except ValueError:
            logger.warning("Default country not found in list, using first country")
            return 0

    def _validate_profile_inputs(self, name: str, age: int, weight: float, 
                               height: float, goals: str) -> bool:
        """Validate profile inputs before saving"""
        try:
            if len(name.strip()) < 3:
                st.error("Name must be at least 3 characters long")
                return False
            
            if len(goals.strip()) < 5:
                st.error("Please provide more detailed fitness goals (at least 5 characters)")
                return False
            
            if age < 13:
                st.error("Age must be at least 13 years")
                return False
            
            if weight < 20 or weight > 300:
                st.error("Please enter a valid weight between 20 and 300 kg")
                return False
            
            if height < 50 or height > 300:
                st.error("Please enter a valid height between 50 and 300 cm")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating profile inputs: {str(e)}")
            st.error("Error validating inputs. Please check your entries.")
            return False

    def _save_profile(self, profile_data: dict) -> bool:
        """Save profile data with error handling"""
        try:
            info = UserInfo(**profile_data)
            st.session_state.user_info = info
            st.session_state.profile_completed = True
            
            # Log the profile update
            log_user_interaction('profile_update', {
                'age': profile_data['age'],
                'country': profile_data['country']
            })
            
            logger.info(f"Successfully saved profile for user: {profile_data['name']}")
            return True
            
        except ValidationError as e:
            logger.error(f"Profile validation error: {str(e)}")
            st.error(f"Please check your inputs: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            st.error("An unexpected error occurred while saving your profile. Please try again.")
            return False

    def display(self, config) -> None:
        """Render profile tab content with improved error handling and validation"""
        try:
            st.header("Your Profile")
            st.write("Please fill this form before moving on.")
            with st.form("profile_form"):
                # Get current values from session state
                current_info = st.session_state.get('user_info', None)
                
                # Basic Information
                name = st.text_input(
                    "Name",
                    value=current_info.name if current_info else "",
                    help="Enter your full name (3-30 characters)",
                    max_chars=30
                )
                
                # Create two columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input(
                        "Age",
                        min_value=config.MIN_AGE,
                        max_value=120,
                        value=current_info.age if current_info else config.DEFAULT_AGE,
                        help="Must be at least 13 years old"
                    )
                    
                    weight = st.number_input(
                        "Weight (kg)",
                        min_value=config.MIN_WEIGHT,
                        max_value=300.0,
                        value=current_info.weight if current_info else config.DEFAULT_WEIGHT,
                        help="Enter weight in kilograms"
                    )
                
                with col2:
                    sex = st.selectbox(
                        "Sex",
                        options=["Male", "Female", "Other"],
                        index=["Male", "Female", "Other"].index(current_info.sex) if current_info else 0
                    )
                    
                    height = st.number_input(
                        "Height (cm)",
                        min_value=config.MIN_HEIGHT,
                        max_value=300.0,
                        value=current_info.height if current_info else config.DEFAULT_HEIGHT,
                        help="Enter height in centimeters"
                    )
                
                # Get and display country selection
                countries = self._get_countries()
                country = st.selectbox(
                    "Country",
                    options=countries,
                    index=self._get_default_country_index(countries)
                )
                
                # Fitness Goals
                goals = st.text_area(
                    "Fitness Goals",
                    value=current_info.goals if current_info else "",
                    help="Describe your fitness goals (minimum 5 characters)",
                    placeholder="Example: Build muscle mass, improve endurance, lose weight",
                    max_chars=500
                )
                
                # Submit button
                submit = st.form_submit_button("Save Profile")
                
                if submit:
                    # Validate inputs
                    if self._validate_profile_inputs(name, age, weight, height, goals):
                        # Prepare profile data
                        profile_data = {
                            "name": name.strip(),
                            "age": age,
                            "sex": sex,
                            "weight": weight,
                            "height": height,
                            "goals": goals.strip(),
                            "country": country
                        }
                        
                        # Save profile
                        if self._save_profile(profile_data):
                            st.success(f"Profile saved successfully! 🎉. Welcome {profile_data['name']} !!!")
                            st.balloons()
                            
                            # Log successful profile creation
                            logger.info(f"Profile created/updated successfully for user: {name}")
                            
                            # Prompt user to check other features
                            st.info("Now you can explore other features of the app!")
                
                            
        except Exception as e:
            logger.error(f"Error displaying profile form: {str(e)}", exc_info=True)
            st.error("An error occurred while loading the profile form. Please refresh the page.")