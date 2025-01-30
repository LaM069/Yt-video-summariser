from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import re
from typing import Dict, List, Optional, Generator, Any
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
import asyncio

@dataclass
class SummaryContext:
    purpose: str
    audience: str
    formality: int  # 1-5 scale
    detail_level: int  # 1-5 scale

class YouTubeSummarizer:
    def __init__(self, ollama_host):
        """Initialize the summarizer with Ollama host URL."""
        self.ollama_host = ollama_host
        self.client = Client(host=ollama_host)
        self._cache = {}
        
    def _generate_cache_key(self, data: str) -> str:
        """Generate a unique cache key for the given data."""
        return hashlib.md5(data.encode()).hexdigest()

    @lru_cache(maxsize=100)
    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL with caching."""
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def fetch_transcript(self, video_id: str) -> str:
        """Fetch and format the video transcript with caching."""
        cache_key = f"transcript_{video_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            formatted_transcript = formatter.format_transcript(transcript)
            self._cache[cache_key] = formatted_transcript
            return formatted_transcript
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")

    def split_transcript(self, transcript: str, content_density: Optional[float] = None) -> List:
        """Split transcript with dynamic chunk sizing based on content density."""
        cache_key = f"chunks_{self._generate_cache_key(transcript)}_{content_density}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculate base chunk size based on content density
        base_chunk_size = 2048
        if content_density:
            # Adjust chunk size inversely to content density
            chunk_size = int(base_chunk_size * (1 / content_density))
            chunk_size = max(1024, min(chunk_size, 4096))  # Keep within reasonable bounds
        else:
            chunk_size = base_chunk_size

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        chunks = text_splitter.create_documents([transcript])
        self._cache[cache_key] = chunks
        return chunks

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            return [model["name"] for model in self.client.list()["models"]]
        except Exception as e:
            raise Exception(f"Error fetching models: {str(e)}")

    def get_summary_styles(self) -> Dict[str, str]:
        """Get available summary styles with descriptions."""
        return {
            "Detailed Summary": "Comprehensive coverage of main points and supporting details",
            "Concise Points": "Brief overview of key information in 2-3 sentences",
            "Key Takeaways": "Essential points in bullet-point format",
            "Executive Brief": "High-level overview focused on business implications",
            "Academic Analysis": "Scholarly analysis with citations and theoretical connections",
            "ELI5 (Explain Like I'm 5)": "Simple explanation suitable for children",
            "Technical Deep-Dive": "Detailed technical analysis for experts",
            "Quick Review": "Ultra-concise summary for rapid review",
            "Study Guide": "Structured summary optimized for learning and retention",
            "Discussion Points": "Key topics and questions for group discussion"
        }

    def get_summary_prompt(self, style: str, context: Optional[SummaryContext] = None) -> str:
        """Get context-aware prompt template based on summary style and context."""
        base_prompts = {
            "Detailed Summary": """Create a detailed summary of this video segment. Focus on the main points 
                               and supporting details: {chunk}""",
            "Concise Points": """Create a concise summary of the key points from this video segment. 
                             Format as 2-3 clear sentences: {chunk}""",
            "Key Takeaways": """Extract the 1-2 most important takeaways from this video segment. 
                            Format as brief, clear bullet points: {chunk}""",
            "Executive Brief": """Summarize this content focusing on business implications and strategic insights: {chunk}""",
            "Academic Analysis": """Analyze this content from an academic perspective, noting theoretical frameworks 
                                and potential research connections: {chunk}""",
            "ELI5": """Explain this content in very simple terms that a child could understand: {chunk}""",
            "Technical Deep-Dive": """Provide a detailed technical analysis of this content, focusing on 
                                  implementation details and technical concepts: {chunk}""",
            "Quick Review": """Provide an ultra-concise summary of only the most critical points: {chunk}""",
            "Study Guide": """Create a study guide format summary with key concepts, definitions, and 
                          important points to remember: {chunk}""",
            "Discussion Points": """Generate discussion points and thought-provoking questions based on 
                                this content: {chunk}"""
        }

        if context:
            # Modify prompt based on context
            purpose_prefix = f"For {context.purpose} purposes: "
            audience_adjustment = f"Adjust explanation for {context.audience} audience. "
            formality_guide = f"Use {'formal' if context.formality > 3 else 'casual'} language. "
            detail_guide = f"Provide {'detailed' if context.detail_level > 3 else 'high-level'} information. "
            
            return purpose_prefix + audience_adjustment + formality_guide + detail_guide + base_prompts.get(style, base_prompts["Detailed Summary"])
        
        return base_prompts.get(style, base_prompts["Detailed Summary"])

    def calculate_content_density(self, text: str) -> float:
        """Calculate content density based on text complexity markers."""
        # Simple density calculation based on average sentence length and unique words
        sentences = text.split('.')
        words = text.split()
        unique_words = len(set(words))
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Normalize to a 0-1 scale
        density = (avg_sentence_length * 0.5 + (unique_words / len(words)) * 0.5) / 10
        return min(max(density, 0.5), 2.0)  # Bound between 0.5 and 2.0

    def create_summary_chain(self, model_name: str, style: str, context: Optional[SummaryContext] = None):
        """Create and return a summary chain that can be used for streaming."""
        llm = ChatOllama(
            model=model_name,
            temperature=0.5,
            base_url=self.ollama_host,
            streaming=True  # Enable streaming
        )
        prompt = ChatPromptTemplate.from_template(self.get_summary_prompt(style, context))
        chain = prompt | llm | StrOutputParser()
        return chain

    async def summarize_stream(
        self,
        chunks: List,
        model_name: str,
        style: str = "Detailed Summary",
        context: Optional[SummaryContext] = None,
    ) -> Generator[str, None, None]:
        """Generate streaming summary using the selected model and style."""
        try:
            chain = self.create_summary_chain(model_name, style, context)
            
            for chunk in chunks:
                density = self.calculate_content_density(chunk.page_content)
                if density > 1.5:
                    subchunks = self.split_transcript(chunk.page_content, density)
                    for subchunk in subchunks:
                        async for token in chain.astream({"chunk": subchunk.page_content}):
                            yield token
                else:
                    async for token in chain.astream({"chunk": chunk.page_content}):
                        yield token
                yield "\n\n"  # Add separation between chunk summaries

            # If style requires final processing
            if len(chunks) > 3 and style not in ["Key Takeaways", "Quick Review"]:
                final_prompt = ChatPromptTemplate.from_template(
                    "Synthesize these summary points into a coherent summary: {text}"
                )
                final_chain = final_prompt | ChatOllama(
                    model=model_name,
                    temperature=0.5,
                    base_url=self.ollama_host,
                    streaming=True
                ) | StrOutputParser()
                
                async for token in final_chain.astream({"text": " ".join(str(c.page_content) for c in chunks)}):
                    yield token

        except Exception as e:
            raise Exception(f"Error during streaming summarization: {str(e)}")

    def summarize(
        self,
        chunks: List,
        model_name: str,
        style: str = "Detailed Summary",
        context: Optional[SummaryContext] = None,
    ) -> str:
        """Generate context-aware summary using the selected model and style (non-streaming)."""
        cache_key = f"summary_{model_name}_{style}_{self._generate_cache_key(str(chunks))}"
        if cache_key in self._cache and not context:  # Don't cache context-specific summaries
            return self._cache[cache_key]

        try:
            llm = ChatOllama(
                model=model_name,
                temperature=0.5,
                base_url=self.ollama_host
            )
            prompt = ChatPromptTemplate.from_template(self.get_summary_prompt(style, context))
            chain = prompt | llm | StrOutputParser()

            # Process chunks and combine summaries
            summaries = []
            for chunk in chunks:
                density = self.calculate_content_density(chunk.page_content)
                if density > 1.5:  # If content is very dense, break into smaller chunks
                    subchunks = self.split_transcript(chunk.page_content, density)
                    for subchunk in subchunks:
                        summary = chain.invoke({"chunk": subchunk.page_content})
                        summaries.append(summary)
                else:
                    summary = chain.invoke({"chunk": chunk.page_content})
                    summaries.append(summary)

            # Combine summaries based on style
            if style == "Key Takeaways":
                final_summary = "\n".join(f"• {summary.strip()}" for summary in summaries)
            else:
                final_summary = " ".join(summaries)

            # For longer content, create a meta-summary
            if len(summaries) > 3 and style not in ["Key Takeaways", "Quick Review"]:
                final_prompt = ChatPromptTemplate.from_template(
                    "Synthesize these summary points into a coherent summary: {text}"
                )
                final_chain = final_prompt | llm | StrOutputParser()
                final_summary = final_chain.invoke({"text": final_summary})

            if not context:  # Only cache non-context-specific summaries
                self._cache[cache_key] = final_summary
            return final_summary

        except Exception as e:
            raise Exception(f"Error during summarization: {str(e)}")