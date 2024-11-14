# # supabase_handler.py
# from datetime import datetime
# from typing import Any
# from supabase import create_client
# import streamlit as st
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class SupabaseHandler:
#     def __init__(self):
#         self.supabase_url = os.getenv("SUPABASE_URL")
#         self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
#         self.client = create_client(self.supabase_url, self.supabase_key)

#     def sign_up(self, email: str, password: str) -> dict[str, Any]:
#         """Register a new user"""
#         try:
#             response = self.client.auth.sign_up({
#                 "email": email,
#                 "password": password
#             })
#             return {"success": True, "user": response.user}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

#     def sign_in(self, email: str, password: str) -> dict[str, Any]:
#         """Sign in an existing user"""
#         try:
#             response = self.client.auth.sign_in_with_password({
#                 "email": email,
#                 "password": password
#             })
#             return {"success": True, "session": response.session}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

#     def sign_out(self) -> dict[str, Any]:
#         """Sign out the current user"""
#         try:
#             self.client.auth.sign_out()
#             return {"success": True}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

#     def get_user(self) -> dict[str, Any] | None:
#         """Get the current user's session"""
#         try:
#             return self.client.auth.get_user()
#         except Exception:
#             return None

#     def save_user_profile(self, user_id: str, profile_data: dict[str, Any]) -> dict[str, Any]:
#         """Save user profile data to Supabase"""
#         try:
#             response = self.client.table('user_profiles').upsert({
#                 'user_id': user_id,
#                 **profile_data
#             }).execute()
#             return {"success": True, "data": response.data}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

#     def get_user_profile(self, user_id: str) -> dict[str, Any]:
#         """Retrieve user profile data from Supabase"""
#         try:
#             response = self.client.table('user_profiles').select("*").eq('user_id', user_id).execute()
#             return {"success": True, "data": response.data[0] if response.data else None}
#         except Exception as e:
#             return {"success": False, "error": str(e)}
    
#     def save_progress_data(self, user_id: str, progress_data: dict[str, Any]) -> dict[str, Any]:
#         """Save progress journal entry to Supabase"""
#         try:
#             # Create a copy of the data to avoid modifying the original
#             data_to_save = progress_data.copy()
            
#             # Ensure datetime is serialized to ISO format string
#             if isinstance(data_to_save.get('date'), datetime):
#                 data_to_save['date'] = data_to_save['date'].isoformat()
                
#             # Insert data into Supabase
#             response = self.client.table('progress_data').insert({
#                 'user_id': user_id,
#                 **data_to_save
#             }).execute()
            
#             return {"success": True, "data": response.data}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

#     def get_progress_data(self, user_id: str) -> dict[str, Any]:
#         """Retrieve user's progress data from Supabase"""
#         try:
#             response = self.client.table('progress_data').select("*").eq('user_id', user_id).execute()
#             return {"success": True, "data": response.data}
#         except Exception as e:
#             return {"success": False, "error": str(e)}

# class AuthUI:
#     def __init__(self, supabase_handler: SupabaseHandler):
#         self.supabase = supabase_handler

#     def render_auth_ui(self):
#         """Render authentication UI"""
#         if 'authenticated' not in st.session_state:
#             st.session_state.authenticated = False

#         if not st.session_state.authenticated:
#             tab1, tab2 = st.tabs(["Login", "Sign Up"])

#             with tab1:
#                 with st.form("login_form"):
#                     email = st.text_input("Email")
#                     password = st.text_input("Password", type="password")
#                     submit = st.form_submit_button("Login")

#                     if submit:
#                         result = self.supabase.sign_in(email, password)
#                         if result["success"]:
#                             st.session_state.authenticated = True
#                             st.session_state.user_id = result["session"].user.id
#                             st.success("Successfully logged in!")
#                             st.rerun()
#                         else:
#                             st.error(f"Login failed: {result['error']}")

#             with tab2:
#                 with st.form("signup_form"):
#                     email = st.text_input("Email")
#                     password = st.text_input("Password", type="password")
#                     password_confirm = st.text_input("Confirm Password", type="password")
#                     submit = st.form_submit_button("Sign Up")

#                     if submit:
#                         if password != password_confirm:
#                             st.error("Passwords do not match!")
#                         else:
#                             result = self.supabase.sign_up(email, password)
#                             if result["success"]:
#                                 st.success("Successfully signed up! Please login.")
#                             else:
#                                 st.error(f"Sign up failed: {result['error']}")

#         else:
#             if st.sidebar.button("Logout"):
#                 result = self.supabase.sign_out()
#                 if result["success"]:
#                     st.session_state.authenticated = False
#                     st.session_state.user_id = None
#                     st.rerun()
#                 else:
#                     st.error(f"Logout failed: {result['error']}")