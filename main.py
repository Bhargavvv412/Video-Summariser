import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

st.title("Video Summarizer Agent")
st.header("This app is powered by Gemini")

def initialize_agent():
    return Agent(
        name = "Video Summerizer Agent",
        model = Gemini(id="gemini-2.5-flash"),
        tools=[DuckDuckGo()],
        markdown= True
    )

multimodel_Agent = initialize_agent()

video_file = st.file_uploader(
    "Upload Video file",type=['mp4','mov','avi'],help="upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path,format="video/mp4",start_time=0)

    user_query = st.text_area(
    "What insights are you seeking from the video?",
    placeholder="Ask anything about the video content. The AI agent will analyze and gather additional content",
    help="Provide specific questions or insights you want from the video."
    )


    if st.button("Analyze Video",key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question and gatering insights...")
        else:
            try:
                with st.spinner("Processing video and gathing insights "):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_propmt = (
                        f"""
                        Analyze the uploaded video for content context.
                        Respond to the following query video insight and supplementry web research:
                        {user_query}
                        Provide a detailed user-friendly and actionable response.
                        """
                    )

                    response = multimodel_Agent.run(analysis_propmt,videos=[processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)
            
            except Exception as e:
                st.error(f"{e}")
            finally:
                Path(video_path).unlink(missing_ok=True)

    else:
        st.info("Upload a video file to begin analysis")


st.markdown("""
<style>
            .stTextArea teaxtarea{
            height: 100px;
            }
            </style>
            """,
            unsafe_allow_html=True
            )