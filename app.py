import streamlit as st
import re
import asyncio
from main import YouTubeSummarizer, SummaryContext

def is_valid_youtube_url(url):
    youtube_regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})'
    return bool(re.match(youtube_regex, url))

async def process_streaming_summary(summarizer, chunks, selected_model, selected_style, context):
    """Process the streaming summary and accumulate tokens."""
    summary = ""
    async for token in summarizer.summarize_stream(
        chunks,
        selected_model,
        selected_style,
        context=context
    ):
        summary += token
        yield summary

def main():
    st.title("YouTube Video Summarizer")
    
    # Session state for storing summary
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    
    # Initialize the summarizer
    summarizer = YouTubeSummarizer(ollama_host='http://localhost:11434')

    # Get available models and styles
    try:
        models = summarizer.get_available_models()
        summary_styles = summarizer.get_summary_styles()
    except Exception as e:
        st.error(f"Error connecting to Ollama server: {str(e)}")
        return

    # User inputs
    url = st.text_input("Enter YouTube URL")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select Model", models)
    with col2:
        selected_style = st.selectbox("Summary Style", list(summary_styles.keys()))
    
    # Advanced options
    with st.expander("Advanced Options"):
        purpose = st.selectbox(
            "Summary Purpose",
            ["General Understanding", "Research", "Study/Exam Prep", "Teaching", "Business/Professional"]
        )
        
        audience = st.selectbox(
            "Target Audience",
            ["General", "Children", "Students", "Professionals", "Experts"]
        )
        
        formality = st.slider(
            "Formality Level",
            1, 5, 3,
            help="1: Very Casual, 5: Highly Formal"
        )
        
        detail_level = st.slider(
            "Detail Level",
            1, 5, 3,
            help="1: High-level overview, 5: In-depth analysis"
        )

    # Streaming options
    use_streaming = st.checkbox("Enable streaming response", value=True)

    if st.button("Generate Summary"):
        if not url:
            st.warning("Please enter a YouTube URL")
            return
        
        if not is_valid_youtube_url(url):
            st.error("Invalid YouTube URL")
            return

        try:
            progress_container = st.empty()
            summary_container = st.empty()
            
            # Extract transcript
            progress_container.info("Extracting transcript...")
            video_id = summarizer.get_video_id(url)
            transcript = summarizer.fetch_transcript(video_id)

            # Create context object
            context = SummaryContext(
                purpose=purpose,
                audience=audience,
                formality=formality,
                detail_level=detail_level
            )

            # Split transcript into chunks
            progress_container.info("Processing content...")
            chunks = summarizer.split_transcript(transcript)
            
            # Clear progress indicator
            progress_container.empty()

            if use_streaming:
                # Streaming mode
                summary_placeholder = st.empty()
                st.session_state.summary = ""
                
                # Create and run async loop for streaming
                async def run_streaming():
                    async for summary in process_streaming_summary(
                        summarizer,
                        chunks,
                        selected_model,
                        selected_style,
                        context
                    ):
                        summary_placeholder.markdown(summary)
                        st.session_state.summary = summary

                # Run the async function
                asyncio.run(run_streaming())
            else:
                # Non-streaming mode
                progress_container.info("Generating summary...")
                summary = summarizer.summarize(
                    chunks,
                    selected_model,
                    selected_style,
                    context=context
                )
                st.session_state.summary = summary
                summary_container.markdown(summary)

            # Show completion message
            st.success("Summary generated!")

            # Show transcript in expander
            with st.expander("View Full Transcript"):
                st.text(transcript)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()