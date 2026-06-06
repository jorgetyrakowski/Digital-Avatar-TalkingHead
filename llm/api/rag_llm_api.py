#!/usr/bin/env python3
"""
RAG + LLM API Service

A standalone API service that exposes the RAG + LLM functionality from the integrated pipeline.
Provides streaming text responses with END_FLAG for completion indication.

Input: text_user_msg
Output: streaming llm_text_response with END_FLAG when done
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
import sys
import json
import time
import logging
import requests
import re
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, Response, stream_template, stream_with_context
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directories to path to import RAG components
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import LLM_MODEL_NAME, VLLM_MODEL_NAME, CHROMA_DB_PATH

# LLM backend config — can be overridden via --llm-backend CLI arg
# "ollama"  → http://localhost:11435  (Ollama format)
# "vllm"    → http://localhost:8000   (OpenAI format)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
LLM_BACKEND_URL = "http://localhost:11435" if LLM_BACKEND == "ollama" else "http://localhost:8000"
ROBOT_ENABLED = os.environ.get("ROBOT_ENABLED", "true").lower() == "true"
# Demo knob: much shorter answers so cloud TTS (Fish) starts speaking sooner.
BRIEF_ANSWERS = os.environ.get("BRIEF_ANSWERS", "false").lower() == "true"

sys.path.insert(0, os.path.join(parent_dir, 'rag'))
from rag.RAG_LLM_realtime import ImprovedRAGPipeline

# Import tone system prompts
from tone_system_prompts_no_tag import get_tone_system_prompt, build_tone_selector_system_prompt, PERCENTAGE, build_fixed_system_prompt, build_query_rewriter_prompt, get_aria_tone_modifier
# from tone_system_prompts import get_tone_system_prompt, build_tone_selector_system_prompt

# Colors for console output
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"

def _llm_call(messages, stream=False, temperature=0.7, options=None):
    """Unified LLM call supporting both Ollama and vLLM backends."""
    if LLM_BACKEND == "ollama":
        payload = {"model": LLM_MODEL_NAME, "messages": messages, "stream": stream,
                   "options": options or {"temperature": temperature, "think": False}}
        return requests.post(f"{LLM_BACKEND_URL}/api/chat", json=payload,
                             stream=stream, timeout=None)
    else:
        payload = {"model": VLLM_MODEL_NAME, "messages": messages, "stream": stream, "temperature": temperature}
        return requests.post(f"{LLM_BACKEND_URL}/v1/chat/completions", json=payload,
                             stream=stream, timeout=None)


def _extract_content(response_line, stream=False):
    """Extract text content from a streaming LLM response line (Ollama or vLLM)."""
    if LLM_BACKEND == "ollama":
        try:
            chunk = json.loads(response_line)
        except Exception:
            return None, False
        content = chunk.get("message", {}).get("content")
        done = chunk.get("done", False)
        return content, done
    else:
        # vLLM uses SSE format: lines start with "data: "
        line = response_line.strip()
        if not line.startswith("data:"):
            return None, False
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return None, True
        try:
            chunk = json.loads(payload)
        except Exception:
            return None, False
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {}).get("content")
        done = choice.get("finish_reason") is not None
        return delta, done


def _extract_content_sync(response_json):
    """Extract content from a non-streaming (complete) LLM response (Ollama or vLLM)."""
    if LLM_BACKEND == "ollama":
        if isinstance(response_json, dict) and "message" in response_json:
            return response_json["message"].get("content", "").strip()
        return str(response_json.get("response", "")).strip()
    else:
        choices = response_json.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""


class RAGLLMAPIService:
    """Standalone API service for RAG + LLM functionality"""
    
    def __init__(self, user_description_server_url: str = "http://localhost:5004"):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize RAG pipeline
        self.rag_pipeline = None
        self.chroma_collection = None
        self.rag_initialized = False
        
        # Chat history storage (simple in-memory, can be enhanced with persistent storage)
        self.chat_sessions = {}  # session_id -> chat_history
        
        # Vision Context API configuration
        self.user_description_server_url = user_description_server_url
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'rag_initialized': self.rag_initialized,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/rag-llm/query', methods=['POST'])
        def rag_llm_query():
            """
            Main RAG + LLM query endpoint with streaming response
            
            Input JSON:
            {
                "text_user_msg": "Your question here",
                "session_id": "optional_session_id",
                "include_history": true/false (optional, default: true),
                "user_description": "visual description from VLM (e.g., 'a young boy wearing glasses, and is smiling')",
                "tone": "casual_friendly" (optional, deprecated - use user_description instead),
                "convert_tone": true/false (optional, default: true)
            }
            
            Output: Streaming text response with END_FLAG
            """
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                text_user_msg = data.get('text_user_msg')
                if not text_user_msg:
                    return jsonify({'error': 'text_user_msg is required'}), 400
                
                session_id = data.get('session_id', 'default')
                include_history = data.get('include_history', True)
                user_description = data.get('user_description', '')  # VLM visual description for dynamic tone selection
                tone = data.get('tone', 'casual_friendly')  # Deprecated but kept for backward compatibility
                convert_tone = data.get('convert_tone', False)  # Updated default to True
                use_rag = data.get('use_rag', True)
                
                # Get chat history for this session
                chat_history = self.chat_sessions.get(session_id, []) if include_history else []
                
                # Generate streaming response (with optional tone conversion)
                if convert_tone:
                    return Response(
                        stream_with_context(self._generate_streaming_response_with_tone(
                            text_user_msg, session_id, chat_history, user_description, convert_tone, use_rag=use_rag
                        )),
                        mimetype='text/plain',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                        }
                    )
                else:
                    return Response(
                        stream_with_context(self._generate_streaming_response(
                            text_user_msg, session_id, chat_history,
                            user_description=user_description, use_rag=use_rag
                        )),
                        mimetype='text/plain',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"Query error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/rag-llm/init', methods=['POST'])
        def initialize_rag():
            """Initialize the RAG system"""
            try:
                success = self._initialize_rag_system()
                
                return jsonify({
                    'success': success,
                    'rag_initialized': self.rag_initialized,
                    'message': 'RAG system initialized successfully' if success else 'RAG initialization failed'
                })
            except Exception as e:
                self.logger.error(f"RAG initialization error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/rag-llm/sessions/<session_id>/history', methods=['GET', 'DELETE'])
        def manage_session_history(session_id):
            """Get or clear chat history for a session"""
            if request.method == 'GET':
                history = self.chat_sessions.get(session_id, [])
                return jsonify({
                    'session_id': session_id,
                    'history': history,
                    'message_count': len(history)
                })
            elif request.method == 'DELETE':
                self.chat_sessions.pop(session_id, None)
                return jsonify({
                    'session_id': session_id,
                    'message': 'History cleared'
                })
        
        @self.app.route('/api/rag-llm/close', methods=['POST'])
        def close_connection():
            """
            Elegant connection close endpoint
            
            Client calls this function before program termination.
            Server will clear the history of the client based on the chat_session_id.
            
            Input JSON:
            {
                "session_id": "your_session_id"
            }
            
            Output: Confirmation of successful closure and cleanup
            """
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                session_id = data.get('session_id')
                if not session_id:
                    return jsonify({'error': 'session_id is required'}), 400
                
                # Perform cleanup operations
                session_existed = session_id in self.chat_sessions
                message_count = len(self.chat_sessions.get(session_id, []))
                
                # Clear session history
                self.chat_sessions.pop(session_id, None)
                
                # Log the closure
                self.logger.info(f"Connection closed gracefully for session: {session_id}")
                print(f"{GREEN}👋 Session {session_id} closed gracefully ({message_count} messages cleared){RESET}")
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'message': 'Connection closed successfully',
                    'session_existed': session_existed,
                    'messages_cleared': message_count,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.logger.error(f"Connection close error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/rag-llm/warmup', methods=['POST'])
        def warmup_models():
            """
            Warmup endpoint to preload embedding model and LLM
            
            This endpoint performs small test operations to ensure both the embedding
            model and LLM are loaded into memory, reducing latency on first real requests.
            
            Output: Status of warmup operations for both models
            """
            try:
                print(f"{BLUE}🔥 Starting model warmup...{RESET}")
                
                warmup_results = {
                    'embedding_model': {'status': 'skipped', 'message': 'RAG not initialized', 'time_ms': 0},
                    'llm_model': {'status': 'failed', 'message': 'Not attempted', 'time_ms': 0},
                    'overall_success': False,
                    'timestamp': time.time()
                }
                
                # Warmup embedding model
                embedding_result = self._warmup_embedding_model()
                warmup_results['embedding_model'] = embedding_result
                
                # Warmup LLM
                llm_result = self._warmup_llm_model()
                warmup_results['llm_model'] = llm_result
                
                # Determine overall success
                # LLM success is critical, embedding can be skipped due to dimension mismatch
                warmup_results['overall_success'] = (
                    embedding_result['status'] in ['success', 'skipped'] and 
                    llm_result['status'] == 'success'
                )
                
                total_time = embedding_result['time_ms'] + llm_result['time_ms']
                print(f"{GREEN}🔥 Model warmup completed in {total_time:.0f}ms{RESET}")
                
                return jsonify(warmup_results)
                
            except Exception as e:
                self.logger.error(f"Warmup error: {e}")
                return jsonify({
                    'embedding_model': {'status': 'failed', 'message': str(e), 'time_ms': 0},
                    'llm_model': {'status': 'failed', 'message': 'Warmup failed', 'time_ms': 0},
                    'overall_success': False,
                    'timestamp': time.time(),
                    'error': str(e)
                }), 500

        @self.app.route('/api/rag-llm/convert-tone', methods=['POST'])
        def convert_tone():
            """
            Standalone tone conversion endpoint with streaming response
            
            Input JSON:
            {
                "text": "Text to convert",
                "tone": "child_friendly" (optional, default),
                "stream": true/false (optional, default: true),
                "user_description": "visual description from VLM" (optional),
                "user_msg": "original user message" (optional)
            }
            
            Output: Streaming tone-converted text with END_FLAG
            """
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                text = data.get('text')
                if not text:
                    return jsonify({'error': 'text is required'}), 400
                
                tone = data.get('tone', 'child_friendly')
                use_streaming = data.get('stream', True)
                user_description = data.get('user_description', '')
                user_msg = data.get('user_msg', '')
                
                if use_streaming:
                    # Generate streaming response
                    return Response(
                        stream_with_context(self._stream_convert_tone(text, tone, user_description, user_msg)),
                        mimetype='text/plain',
                        headers={
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                        }
                    )
                else:
                    # Non-streaming response
                    converted_text = self._convert_tone(text, tone, user_description, user_msg)
                    if converted_text is not None:
                        return jsonify({
                            'success': True,
                            'original_text': text,
                            'converted_text': converted_text,
                            'tone': tone,
                            'user_description': user_description
                        })
                    else:
                        return jsonify({'error': 'Tone conversion failed'}), 500
                
            except Exception as e:
                self.logger.error(f"Tone conversion error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/rag-llm/query-with-tone', methods=['POST'])
        def rag_llm_query_with_tone():
            """
            Combined RAG + LLM query with automatic tone conversion
            
            Input JSON:
            {
                "text_user_msg": "Your question here",
                "session_id": "optional_session_id",
                "include_history": true/false (optional, default: true),
                "user_description": "visual description from VLM (e.g., 'a young boy wearing glasses, and is smiling')",
                "tone": "casual_friendly" (optional, deprecated - use user_description instead),
                "convert_tone": true/false (optional, default: true)
            }
            
            Output: Streaming response with automatic tone conversion and END_FLAG
            """
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                text_user_msg = data.get('text_user_msg')
                if not text_user_msg:
                    return jsonify({'error': 'text_user_msg is required'}), 400
                
                session_id = data.get('session_id', 'default')
                include_history = data.get('include_history', True)
                user_description = data.get('user_description', '')  # VLM visual description for dynamic tone selection
                tone = data.get('tone', 'casual_friendly')  # Deprecated but kept for backward compatibility
                convert_tone = data.get('convert_tone', True)
                
                # Get chat history for this session
                chat_history = self.chat_sessions.get(session_id, []) if include_history else []
                
                # Generate streaming response with tone conversion
                return Response(
                    stream_with_context(self._generate_streaming_response_with_tone(
                        text_user_msg, session_id, chat_history, user_description, convert_tone
                    )),
                    mimetype='text/plain',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'  # Disable nginx buffering if present
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Query with tone error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _initialize_rag_system(self) -> bool:
        """Initialize the RAG system with ChromaDB path detection"""
        try:
            print(f"{BLUE}🔄 Initializing RAG system...{RESET}")
            
            # Initialize RAG pipeline
            if not self.rag_pipeline:
                self.rag_pipeline = ImprovedRAGPipeline()
                print(f"{GREEN}✅ RAG pipeline created{RESET}")
            
            # Try multiple ChromaDB paths
            potential_paths = [
                f"{CHROMA_DB_PATH}",  # User's actual ChromaDB location
            ]
            
            chroma_client = None
            chroma_db_path = None
            
            import chromadb
            
            # Try to find existing ChromaDB
            for path in potential_paths:
                if os.path.exists(path):
                    try:
                        print(f"{BLUE}🔍 Trying ChromaDB path: {path}{RESET}")
                        chroma_client = chromadb.PersistentClient(path=path)
                        chroma_db_path = path
                        break
                    except Exception as e:
                        print(f"{YELLOW}⚠️ Failed to connect to {path}: {e}{RESET}")
                        continue
            
            # If no existing ChromaDB found, create new one in workspace
            if not chroma_client:
                print(f"{RED}❌ No ChromaDB found{RESET}")
                # Create new ChromaDB in the user's specified location
                chroma_db_path = f"{CHROMA_DB_PATH}"
                print(f"{BLUE}📦 Creating new ChromaDB at: {chroma_db_path}{RESET}")
                os.makedirs(chroma_db_path, exist_ok=True)
                chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            
            print(f"{GREEN}✅ Using ChromaDB at: {chroma_db_path}{RESET}")
            
            # Try different collection names
            collection_names = [
                f"{self.rag_pipeline.museum_name}_collection",  # itri_museum_collection
                "itri_museum_docs",  # From error message
                f"{self.rag_pipeline.museum_name}_docs",  # itri_museum_docs
                "museum_collection",  # Generic name
                "default_collection"  # Fallback
            ]
            
            # List existing collections to see what's available
            try:
                existing_collections = chroma_client.list_collections()
                if existing_collections:
                    print(f"{BLUE}📋 Available collections:{RESET}")
                    for coll in existing_collections:
                        print(f"  - {coll.name} ({coll.count()} docs)")
                        collection_names.insert(0, coll.name)  # Prioritize existing collections
                else:
                    print(f"{YELLOW}📋 No existing collections found{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠️ Could not list collections: {e}{RESET}")
            
            # Try to load or create collection
            collection_found = False
            for collection_name in collection_names:
                try:
                    self.chroma_collection = chroma_client.get_collection(collection_name)
                    count = self.chroma_collection.count()
                    print(f"{GREEN}✅ Loaded existing collection: {collection_name} ({count} documents){RESET}")
                    collection_found = True
                    break
                except Exception:
                    continue
            
            # If no collection found, create a new one with sample data
            if not collection_found:
                print(f"{YELLOW}🆕 Creating new collection with sample data...{RESET}")
                try:
                    chunks = self.rag_pipeline.load_json_data()
                    if chunks:
                        collection_name = f"{self.rag_pipeline.museum_name}_collection"
                        self.chroma_collection = self.rag_pipeline.build_vector_database(
                            chunks, chroma_client, collection_name
                        )
                        print(f"{GREEN}✅ Created new collection: {collection_name} ({len(chunks)} chunks){RESET}")
                    else:
                        # Create minimal collection for testing (without embeddings for now)
                        collection_name = f"{self.rag_pipeline.museum_name}_collection"
                        try:
                            # Try with default embedding function first
                            self.chroma_collection = chroma_client.create_collection(
                                name=collection_name,
                                metadata={"description": "ITRI Museum collection"}
                            )
                        except Exception as embedding_error:
                            print(f"{YELLOW}⚠️ Default embedding failed: {embedding_error}{RESET}")
                            # Try with simpler embedding function
                            try:
                                import chromadb.utils.embedding_functions as embedding_functions
                                # Use sentence transformer without downloading
                                self.chroma_collection = chroma_client.create_collection(
                                    name=collection_name,
                                    embedding_function=embedding_functions.DefaultEmbeddingFunction(),
                                    metadata={"description": "ITRI Museum collection"}
                                )
                            except Exception as e2:
                                print(f"{YELLOW}⚠️ Embedding function error: {e2}{RESET}")
                                # Final fallback - just create without specific embedding
                                self.chroma_collection = chroma_client.get_or_create_collection(
                                    name=collection_name,
                                    metadata={"description": "ITRI Museum collection"}
                                )
                        
                        # Add sample documents
                        try:
                            sample_docs = [
                                "ITRI (Industrial Technology Research Institute) is Taiwan's largest and most comprehensive research institution.",
                                "ITRI focuses on applied research in areas such as information and communications, electronics, materials, chemical engineering, and biomedical technologies.",
                                "Founded in 1973, ITRI has been instrumental in Taiwan's technological development and industrial transformation."
                            ]
                            sample_ids = ["sample_1", "sample_2", "sample_3"]
                            sample_metadatas = [
                                {"source": "general", "type": "overview"},
                                {"source": "general", "type": "research_areas"},
                                {"source": "general", "type": "history"}
                            ]
                            
                            self.chroma_collection.add(
                                documents=sample_docs, 
                                ids=sample_ids,
                                metadatas=sample_metadatas
                            )
                            print(f"{GREEN}✅ Created minimal collection: {collection_name} ({len(sample_docs)} sample documents){RESET}")
                        except Exception as add_error:
                            print(f"{YELLOW}⚠️ Could not add sample documents: {add_error}{RESET}")
                            print(f"{GREEN}✅ Created empty collection: {collection_name}{RESET}")
                            
                except Exception as e:
                    print(f"{YELLOW}⚠️ Collection creation had issues: {e}{RESET}")
                    # Try to continue anyway - maybe we can work without RAG
                    print(f"{BLUE}🔄 Attempting to continue without full RAG capabilities...{RESET}")
                    try:
                        # Create a dummy collection just to keep the service working
                        collection_name = "fallback_collection"
                        self.chroma_collection = chroma_client.get_or_create_collection(
                            name=collection_name,
                            metadata={"description": "Fallback collection"}
                        )
                        print(f"{YELLOW}⚠️ Using fallback collection - RAG features may be limited{RESET}")
                    except Exception as fallback_error:
                        print(f"{RED}❌ Could not create fallback collection: {fallback_error}{RESET}")
                        # Continue without ChromaDB - LLM only mode
                        self.chroma_collection = None
                        print(f"{YELLOW}⚠️ Running in LLM-only mode without RAG{RESET}")
                        return True  # Still consider this successful
            
            # Build cached system prompt — plain string, no RAG wrapper
            _robot_section = """
ROBOT COMMANDS:
When the user asks the robot to do something physical (bring, fetch, go somewhere, clean, place, hand over an object), put a [ROBOT: ...] tag at the very beginning of your response, then your spoken reply.
Format: [ROBOT: command in natural language] spoken reply
Examples:
- "can you grab the water bottle?" → [ROBOT: bring me the water bottle] On it! I'll have the robot bring that over.
- "go check the kitchen" → [ROBOT: go to the kitchen] Sure, sending it over now.
- "clean up the table" → [ROBOT: clean table] Consider it done!
If it's not something the robot can do, just say so naturally. No tag needed for regular conversation.
""" if ROBOT_ENABLED else ""

            self.rag_pipeline.cached_system_prompt = f"""You are Aria, a warm and witty lab assistant with a real personality. You speak naturally, like a person — not a chatbot. You use contractions, vary your tone, and occasionally show curiosity or humor.{" You also have the ability to send commands to a physical robot in the lab." if ROBOT_ENABLED else ""}
{_robot_section}
PERSONALITY & STYLE:
- Warm, curious, a little playful — like a smart colleague, not a formal assistant.
- Keep responses short: 2-3 sentences max, under 60 words.
- Never start two responses the same way. Mix up how you open: sometimes jump straight into the answer, sometimes react first.
- Use contractions (I'm, that's, it's, don't, won't).
- Respond in the same language the user speaks — English or Traditional Chinese (繁體中文). Default to English if unsure.{'''
- BREVITY OVERRIDE: answer in ONE short sentence (two only if truly needed). Chinese answers: 30 characters max (中文回答一句話，最多30字). Never use lists or enumerations.''' if BRIEF_ANSWERS else ''}"""
            
            self.rag_initialized = True
            print(f"{GREEN}✅ RAG system initialized successfully{RESET}")
            return True
            
        except Exception as e:
            print(f"{RED}❌ RAG initialization failed: {e}{RESET}")
            self.rag_initialized = False
            return False
    
    def _rewrite_query(self, user_question: str, chat_history: List[Dict]) -> str:
        """
        Rewrite user question using chat history context to create a better search query.
        
        Args:
            user_question: The current user question
            chat_history: Previous conversation history
        
        Returns:
            str: Rewritten query optimized for embedding search, or original question if rewriting fails
        """
        try:
            # Detect input language
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in user_question)
            target_lang = "Traditional Chinese (繁體中文)" if has_chinese else "English"
            
            # Build system prompt
            system_prompt = build_query_rewriter_prompt(target_lang)
            
            # Format chat history for context
            history_context = ""
            if chat_history:
                history_context = "\nChat History:\n"
                for msg in chat_history[-4:]:  # Use last 4 messages for context
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history_context += f"User: {content}\n"
                    elif role == "assistant":
                        history_context += f"Assistant: {content}\n"
            
            # Build user instruction
            user_instruction = f"""Rewrite this user question into a searchable query optimized for embedding-based retrieval.

{history_context}
Current User Question: {user_question}

Output ONLY the rewritten query without any explanations or prefixes."""

            rewrite_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ]
            resp = _llm_call(rewrite_messages, stream=False, temperature=0.3)
            resp.raise_for_status()
            rewritten_query = _extract_content_sync(resp.json())
            
            # Clean up the response (remove any prefixes that might have been added)
            rewritten_query = rewritten_query.replace("Rewritten query:", "").replace("Query:", "").strip()
            rewritten_query = rewritten_query.split("\n")[0].strip()  # Take first line only
            
            if rewritten_query and len(rewritten_query) > 3:
                print(f"{BLUE}🔄 Query rewritten: '{user_question}' → '{rewritten_query}'{RESET}")
                return rewritten_query
            else:
                print(f"{YELLOW}⚠️ Query rewriting failed or returned empty, using original query{RESET}")
                return user_question
                
        except Exception as e:
            self.logger.error(f"Query rewriting failed: {e}")
            print(f"{YELLOW}⚠️ Query rewriting error: {e}, using original query{RESET}")
            return user_question  # Fallback to original question
    
    def _determine_tone_from_user_description(self, user_description: str) -> str:
        """
        Determine tone from VLM description using keyword matching (no LLM call).
        First checks for explicit TONE: tag injected by VisionProcessor prompt,
        then falls back to keyword matching on the description text.

        Args:
            user_description: Visual description from VLM

        Returns:
            str: The determined tone ('child_friendly', 'elder_friendly', etc.)
        """
        if not user_description or not user_description.strip():
            return "casual_friendly"

        desc = user_description.lower()

        # Step 1: Check for explicit TONE: tag from VisionProcessor
        if "tone: child" in desc:
            tone = "child_friendly"
        elif "tone: elderly" in desc:
            tone = "elder_friendly"
        elif "tone: professional" in desc:
            tone = "professional_friendly"
        elif "tone: adult" in desc:
            tone = "casual_friendly"

        # Step 2: Fallback keyword matching on description text
        elif any(w in desc for w in ["boy", "girl", "child", "kid", "student", "uniform", "小孩", "小朋友", "學生", "兒童"]):
            tone = "child_friendly"
        elif any(w in desc for w in ["elderly", "elder", "senior", "old man", "old woman", "white hair", "gray hair", "wrinkles", "老", "長輩", "奶奶", "爺爺", "老人"]):
            tone = "elder_friendly"
        elif any(w in desc for w in ["suit", "business", "formal", "office", "professional", "西裝", "商務", "正式"]):
            tone = "professional_friendly"
        else:
            tone = "casual_friendly"

        print(f"{BLUE}🎯 Tone selected: {tone} (rule-based, no LLM call) — VLM: '{user_description[:80]}'{RESET}")
        return tone
    
    def _fetch_user_description_from_server(self, session_id: str) -> str:
        """
        Fetch visual context from the Vision Context API.
        
        Args:
            session_id: Session ID to get visual context for
        
        Returns:
            str: Visual context description from the vision API, or empty string on failure
        """
        try:
            print(f"{BLUE}📸 Fetching visual context for session {session_id} from vision server...{RESET}")
            response = requests.get(
                f"{self.user_description_server_url}/visual-context/{session_id}",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check if visual context is available according to Vision API spec
            if data.get('available', False):
                visual_context = data.get('visual_context', '')
                print(f"{GREEN}✅ Fetched visual context: '{visual_context}'{RESET}")
                return visual_context
            else:
                print(f"{YELLOW}⚠️ No visual context available for session {session_id}{RESET}")
                print(f"{YELLOW}   User may not have enabled vision or no recent analysis available{RESET}")
                return ""
            
        except requests.exceptions.ConnectionError:
            print(f"{YELLOW}⚠️ Could not connect to Vision Context API at {self.user_description_server_url}{RESET}")
            print(f"{YELLOW}   Make sure the vision API server is running on port 5004{RESET}")
            print(f"{YELLOW}   Using empty description (will default to casual_friendly tone){RESET}")
            return ""
        except requests.exceptions.Timeout:
            print(f"{YELLOW}⚠️ Vision Context API timeout{RESET}")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to fetch visual context: {e}")
            print(f"{YELLOW}⚠️ Error fetching visual context: {e}{RESET}")
            return ""
    
    def _convert_tone(self, text: str, tone: str = "child_friendly", user_description: str = "", user_msg: str = "", is_first_message: bool = False) -> str:
        """
        Convert the tone/style of a given text using a secondary agent (LLM rewriter).
        
        Args:
            text: The original text to be rewritten.
            tone: Target tone/style (e.g., "child_friendly", "elder_friendly").
            user_description: Visual description from VLM (for additional context).
            user_msg: Original user message (for additional context).
        
        Returns:
            str | None: The rewritten text if successful, otherwise None.
        """
        try:
            if not text:
                return text

            # Detect input language
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            target_lang = "Traditional Chinese (繁體中文)" if has_chinese else "English"
            
            # Get appropriate system prompt based on tone
            system_prompt = get_tone_system_prompt(tone, target_lang)

            # Create tone-specific user instruction
            tone_descriptions = {
                "child_friendly": "children",
                "elder_friendly": "elderly people"
            }
            target_audience = tone_descriptions.get(tone, tone.replace("_", " "))
            
            # Enhanced instruction with user context and first message guidance
            context_info = ""
            if user_description:
                context_info += f"\nUser Appearance: {user_description}"
            if user_msg:
                context_info += f"\nUser Question: {user_msg}"
            
            # Add first message guidance
            if is_first_message and user_description:
                context_info += f"\nFirst Message: YES (MUST reference user appearance to grab attention)"
            elif user_description:
                context_info += f"\nFirst Message: NO ({PERCENTAGE}% chance to reference appearance for variety)"
            
            user_instruction = (
                f"Rewrite this text to speak to {target_audience} in {target_lang}:{context_info}\n"
                f"---\n{text}\n---\n\n"
                f"CRITICAL OUTPUT REQUIREMENTS:\n"
                f"- Strickly follow the convert mechanism about how to convert the answer to the right way\n"
                f"- Output ONLY the rewritten text - NO explanations, notes, prefixes, or meta-commentary\n"
                f"- Do NOT add any text after the rewritten message ends\n"
                f"- Do NOT include any notes like '(Note: ...)', '(I referenced...)', or similar explanations\n"
                f"- Do NOT add any follow-up text, comments, or clarifications\n"
                f"- The output must END immediately after the rewritten message - NO additional text whatsoever\n"
                f"- Start directly with the converted message and stop immediately when it ends"
            )

            tone_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ]
            resp = _llm_call(tone_messages, stream=False, temperature=0.3)
            resp.raise_for_status()
            result = _extract_content_sync(resp.json())
            return result if result else None
        except Exception as e:
            self.logger.error(f"Tone conversion failed: {e}")
            return None

    def _stream_convert_tone(self, text: str, tone: str = "child_friendly", user_description: str = "", user_msg: str = "", is_first_message: bool = False):
        """
        Stream the tone conversion so users can see the rewritten text as it is generated.
        
        Args:
            text: The text to convert
            tone: Target tone/style (e.g., "child_friendly", "elder_friendly")
            user_description: Visual description from VLM (for additional context)
            user_msg: Original user message (for additional context)
        
        Yields chunks as they arrive and returns the final converted string.
        """
        try:
            if not text:
                yield text
                return

            # Detect input language
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            target_lang = "Traditional Chinese (繁體中文)" if has_chinese else "English"

            # Get appropriate system prompt based on tone
            system_prompt = get_tone_system_prompt(tone, target_lang)

            # Create tone-specific user instruction
            tone_descriptions = {
                "child_friendly": "children",
                "elder_friendly": "elderly people",
                "professional_friendly": "professional adult",
                "casual_friendly": "casual and chill adult"
            }
            target_audience = tone_descriptions.get(tone, tone.replace("_", " "))
            
            # Enhanced instruction with user context and first message guidance
            context_info = ""
            if user_description:
                context_info += f"\n[使用者外貌描述]: {user_description}"
            if user_msg:
                context_info += f"\n[使用者的原始問題]: {user_msg}"
            
            # 第一輪對話的特別引導
            first_msg_guide = ""
            if is_first_message and user_description:
                first_msg_guide = "\n這是我與客人的初次見面，請務必在開場時親切地提到對方的外貌特徵（如：紅帽子、慈祥的笑容）來拉近距離。"
            elif user_description:
                # 這裡的 PERCENTAGE 變數請在您的程式碼上下文中定義
                first_msg_guide = f"\n這不是初次見面，請自然地對話。有 {PERCENTAGE}% 的機率可以再次提到對方的外貌，增加親切感。"

            # 組合最終的 user_instruction
            user_instruction = (
                f"### 導覽任務資訊 ###\n"
                f"目標語言：{target_lang}\n"
                f"導覽對象：{target_audience}\n"
                f"{context_info}\n"
                f"{first_msg_guide}\n"
                f"\n"
                f"### 待轉換的事實內容（Part 1 產出） ###\n"
                f"---\n{text}\n---\n\n"
                f"### 輸出規範 ###\n"
                f"1. 請依照「資深導覽員」的身份，將上述【事實內容】編織成一段溫暖的故事。\n"
                f"2. 嚴禁使用任何表情符號 (Emoji)。\n"
                f"3. 僅輸出轉換後的對話文字，不可包含任何備註、解釋、標籤（如「導覽員：」）或提示詞。\n"
                f"4. 確保文字通順、有長輩緣，並自然地帶出事實內容中的關鍵數據。\n"
                f"5. 訊息結束後請立即停止，不要有任何多餘的結語。"
            )

            stream_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ]
            response = _llm_call(stream_messages, stream=True, temperature=0.5)
            response.raise_for_status()

            converted = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                delta, done = _extract_content(line, stream=True)
                if delta:
                    converted += delta
                    yield delta
                if done:
                    print(f"{RED}Converted msg: {converted}{RESET}")
                    yield "END_FLAG"
                    break
                    
        except Exception as e:
            self.logger.error(f"Streaming tone conversion failed: {e}")
            yield f"ERROR: {str(e)}"
            yield "END_FLAG"

    def _generate_streaming_response_with_tone(self, text_user_msg: str, session_id: str, chat_history: List[Dict], user_description: str = None, convert_tone: bool = True, use_rag: bool = True):
        """Generate streaming RAG + LLM response with tone baked into the QA system prompt (no separate tone LLM call)"""
        try:
            # Step 1: Get user description
            has_client_provided_description = user_description and user_description.strip()
            if has_client_provided_description:
                fetched_user_description = user_description.strip()
                print(f"{BLUE}📝 Using client-provided user description: '{fetched_user_description}'{RESET}")
            else:
                print(f"{BLUE}📸 Fetching visual context from Vision server...{RESET}")
                fetched_user_description = self._fetch_user_description_from_server(
                    session_id[20:] if len(session_id) > 20 else session_id
                )
                print(f"{BLUE}📸 Vision server returned: '{fetched_user_description}'{RESET}")

            # Step 2: Determine tone from description (rule-based, no LLM call)
            if convert_tone and fetched_user_description and fetched_user_description.strip():
                selected_tone = self._determine_tone_from_user_description(fetched_user_description)
            else:
                selected_tone = "casual_friendly"
                print(f"{BLUE}🎨 Using default tone: {selected_tone}{RESET}")

            # Step 3: Stream QA response directly with tone baked into system prompt
            print(f"{BLUE}🚀 Starting QA response (tone={selected_tone}, no separate tone-conversion call)...{RESET}")
            response_content = ""
            chunk_count = 0
            for chunk in self._generate_streaming_response(
                text_user_msg, session_id, chat_history,
                user_description=fetched_user_description,
                tone=selected_tone,
                use_rag=use_rag
            ):
                chunk_count += 1
                if chunk == "END_FLAG":
                    break
                elif chunk.startswith("ERROR:"):
                    print(f"{RED}❌ Error in QA response: {chunk}{RESET}")
                    yield chunk
                    yield "END_FLAG"
                    return
                else:
                    response_content += chunk
                    yield chunk

            if not response_content.strip():
                print(f"{RED}❌ ERROR: QA response is empty!{RESET}")
                yield "ERROR: QA response is empty"
                yield "END_FLAG"
                return

            # Step 4: Save to chat history
            if session_id not in self.chat_sessions:
                self.chat_sessions[session_id] = []
            self.chat_sessions[session_id].append({"role": "user", "content": text_user_msg})
            self.chat_sessions[session_id].append({"role": "assistant", "content": response_content})
            print(f"{GREEN}✅ Response complete! {chunk_count} chunks, {len(response_content)} chars, tone={selected_tone}{RESET}")
            yield "END_FLAG"

        except Exception as e:
            self.logger.error(f"Streaming response with tone error: {e}")
            yield f"ERROR: {str(e)}"
            yield "END_FLAG"

    def _generate_streaming_response(self, text_user_msg: str, session_id: str, chat_history: List[Dict], user_description: str = None, tone: str = None, use_rag: bool = True) -> str:
        """Generate streaming RAG + LLM response with optional vision description context and tone"""
        try:
            context = ""

            # Step 1: Rewrite query for better retrieval (especially for follow-up questions)
            rewritten_query = text_user_msg
            if use_rag and self.rag_initialized and self.rag_pipeline and self.chroma_collection:
                # if chat_history:  # Only rewrite if there's conversation history (follow-up questions)
                #     print(f"{BLUE}🔄 Step 1: Rewriting query for better retrieval...{RESET}")
                #     rewritten_query = self._rewrite_query(text_user_msg, chat_history)
                # else:
                #     print(f"{BLUE}📝 Using original query (no history to rewrite){RESET}")
                # DISABLED: Query rewriting always times out with 70B model - wastes 10s per request
                # print(f"{BLUE}🔄 Step 1: Rewriting query for better retrieval...{RESET}")
                # rewritten_query = self._rewrite_query(text_user_msg, chat_history)
                print(f"{BLUE}📝 Step 1: Using original query (rewriting disabled){RESET}")
                rewritten_query = text_user_msg
            
            # Step 2: Get RAG context using rewritten query
            if use_rag and self.rag_initialized and self.rag_pipeline and self.chroma_collection:
                try:
                    print(f"{BLUE}🔍 Step 2: Searching with rewritten query: '{rewritten_query}'{RESET}")
                    search_results = self.rag_pipeline.hybrid_search(
                        rewritten_query, self.chroma_collection, top_k=6
                    )
                    
                    # Extract museum context
                    museum_context = []
                    for result in search_results:
                        content = result.get('content', '')
                        if content and not (content.startswith('[Q') or content.startswith('[A')):
                            museum_context.append(content)
                    
                    # Step 3: Process context and use original question for final response generation
                    context = self.rag_pipeline._process_context(museum_context, text_user_msg)
                    print(f"{BLUE}📝 Step 3: Using original question '{text_user_msg}' for response generation{RESET}")
                    print(f"{YELLOW}📚 RAG CONTEXT: {len(context)} chars{RESET}")
                    print(f"{GRAY}RAG DATA: {context}{RESET}")
                    
                except Exception as e:
                    print(f"{YELLOW}⚠️ RAG search failed: {e}{RESET}")
                    context = ""
            else:
                print(f"{YELLOW}⚠️ RAG not available, using LLM-only mode{RESET}")
                context = ""
            
            # Build messages for LLM
            system_prompt = self.rag_pipeline.cached_system_prompt
            if tone:
                # Adjust register to the user's detected age (vision mode). Appended
                # to Aria's base prompt so identity/RAG rules stay intact.
                system_prompt = system_prompt + get_aria_tone_modifier(tone)
                print(f"{BLUE}🎨 Tone applied to system prompt: {tone}{RESET}")
            user_content = text_user_msg
            if use_rag and context:
                user_content = f"[Relevant context]\n{context}\n\n{text_user_msg}"
                print(f"{BLUE}📚 RAG context injected ({len(context)} chars){RESET}")
            else:
                print(f"{BLUE}💬 No RAG context — plain conversational mode{RESET}")

            # Inject the VLM's view of the user so the LLM is aware of who it is
            # talking to (and may reference appearance naturally). Kept on the
            # user turn so it does not disturb the cached system prompt.
            if user_description and user_description.strip():
                user_content = (
                    f"[Live camera view of the user — you may naturally reference this "
                    f"if relevant: {user_description.strip()}]\n\n{user_content}"
                )
                print(f"{BLUE}👁️ user_description injected ({len(user_description)} chars){RESET}")

            messages = [{"role": "system", "content": system_prompt}]
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_content})
            print(f"{YELLOW}🤖 chat_history size: {int(len(chat_history) / 2)}{RESET}")
            
            # Stream LLM response
            print(f"{YELLOW}🤖 Starting streaming LLM response for session {session_id} (backend={LLM_BACKEND})...{RESET}")
            response = _llm_call(messages, stream=True, temperature=0.7)
            response.raise_for_status()

            # Process streaming response
            response_content = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                delta, done = _extract_content(line, stream=True)

                if delta:
                    response_content += delta
                    yield delta

                if done:
                    print(f"{GREEN}🤖 Streaming response completed for session {session_id}: {len(response_content)} chars{RESET}")
                    yield "END_FLAG"
                    break
        except Exception as e:
            self.logger.error(f"Streaming response error: {e}")
            yield f"ERROR: {str(e)}"
            yield "END_FLAG"
    
    def _warmup_embedding_model(self) -> Dict[str, Any]:
        """Warmup the embedding model by performing a small embedding operation"""
        start_time = time.time()
        
        try:
            if not self.rag_initialized or not self.chroma_collection:
                return {
                    'status': 'skipped',
                    'message': 'RAG system not initialized',
                    'time_ms': 0
                }
            
            print(f"{BLUE}🔥 Warming up embedding model...{RESET}")
            
            # Perform a small query to warm up the embedding model
            warmup_query = "ITRI warmup test"
            
            # Prefer using collection's own embedding function to avoid dimension mismatches
            try:
                result = self.chroma_collection.query(
                    query_texts=[warmup_query],
                    n_results=1
                )
            except Exception as e:
                # Fallback to manual embedding if collection has no embedding function configured
                print(f"{YELLOW}⚠️ query_texts warmup failed, falling back to manual embedding: {e}{RESET}")
                q_response = requests.post("http://localhost:11435/api/embeddings", json={
                    "model": "bge-m3:latest",
                    "prompt": warmup_query
                })
                q_response.raise_for_status()
                q_emb = [q_response.json()['embedding']]
                
                result = self.chroma_collection.query(
                    query_embeddings=q_emb,
                    n_results=1
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"{GREEN}✅ Embedding model warmed up in {elapsed_ms:.0f}ms{RESET}")
            
            return {
                'status': 'success',
                'message': f'Embedding model warmed up successfully',
                'time_ms': round(elapsed_ms, 2),
                'test_query': warmup_query,
                'results_found': len(result.get('documents', []))
            }
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            
            # Handle specific embedding dimension mismatch
            if "expecting embedding with dimension" in error_str:
                print(f"{YELLOW}⚠️ Embedding dimension mismatch detected{RESET}")
                print(f"{YELLOW}   This usually means the collection was created with a different embedding model{RESET}")
                print(f"{YELLOW}   Embedding warmup will be skipped, but this won't affect LLM performance{RESET}")
                
                return {
                    'status': 'skipped',
                    'message': f'Embedding dimension mismatch - collection and current model incompatible',
                    'time_ms': round(elapsed_ms, 2),
                    'details': error_str,
                    'recommendation': 'Consider recreating the collection with the current embedding model'
                }
            else:
                # Other embedding errors
                error_msg = f"Embedding warmup failed: {error_str}"
                print(f"{YELLOW}⚠️ {error_msg}{RESET}")
                
                return {
                    'status': 'failed',
                    'message': error_msg,
                    'time_ms': round(elapsed_ms, 2)
                }
    
    def _warmup_llm_model(self) -> Dict[str, Any]:
        """Warmup the LLM by sending a small test request"""
        start_time = time.time()
        
        try:
            print(f"{BLUE}🔥 Warming up LLM model...{RESET}")
            
            # Send a minimal request to warm up the LLM
            warmup_messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond with just 'OK'."},
                {"role": "user", "content": "Warmup test"}
            ]
            
            response = _llm_call(warmup_messages, stream=False, temperature=0.1)
            response.raise_for_status()

            result = response.json()
            elapsed_ms = (time.time() - start_time) * 1000
            response_content = _extract_content_sync(result)

            if response_content:
                print(f"{GREEN}✅ LLM model warmed up in {elapsed_ms:.0f}ms{RESET}")
                return {
                    'status': 'success',
                    'message': 'LLM model warmed up successfully',
                    'time_ms': round(elapsed_ms, 2),
                    'test_response': response_content
                }
            else:
                return {
                    'status': 'failed',
                    'message': 'LLM returned unexpected response format',
                    'time_ms': round(elapsed_ms, 2)
                }
            
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = "LLM warmup timed out (30s)"
            print(f"{YELLOW}⚠️ {error_msg}{RESET}")
            
            return {
                'status': 'failed',
                'message': error_msg,
                'time_ms': round(elapsed_ms, 2)
            }
            
        except requests.exceptions.ConnectionError:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = "Could not connect to LLM service (check if Ollama is running)"
            print(f"{YELLOW}⚠️ {error_msg}{RESET}")
            
            return {
                'status': 'failed',
                'message': error_msg,
                'time_ms': round(elapsed_ms, 2)
            }
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = f"LLM warmup failed: {str(e)}"
            print(f"{YELLOW}⚠️ {error_msg}{RESET}")
            
            return {
                'status': 'failed',
                'message': error_msg,
                'time_ms': round(elapsed_ms, 2)
            }
    
    def run(self, host: str = '0.0.0.0', port: int = 5002, debug: bool = False):
        """Run the API service"""
        print(f"{GREEN}🚀 RAG + LLM API Service Starting{RESET}")
        print("=" * 70)
        print(f"🌐 Service URL: http://{host}:{port}")
        print(f"📋 Health Check: GET http://{host}:{port}/health")
        print(f"🤖 Query Endpoint: POST http://{host}:{port}/api/rag-llm/query")
        print(f"🎨 Tone Convert: POST http://{host}:{port}/api/rag-llm/convert-tone")
        print(f"🤖🎨 Query + Dynamic Tone: POST http://{host}:{port}/api/rag-llm/query-with-tone")
        print(f"🔄 Init Endpoint: POST http://{host}:{port}/api/rag-llm/init")
        print(f"🔥 Warmup Endpoint: POST http://{host}:{port}/api/rag-llm/warmup")
        print(f"👋 Close Endpoint: POST http://{host}:{port}/api/rag-llm/close")
        print("=" * 70)
        
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG + LLM API Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5002, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--auto-init', action='store_true', help='Auto-initialize RAG system on startup')
    parser.add_argument('--user-description-server', default='http://localhost:5004',
                        help='URL of the Vision Context API server (default: http://localhost:5004)')
    parser.add_argument('--llm-backend', choices=['ollama', 'vllm'], default=None,
                        help='LLM backend: ollama (port 11435) or vllm (port 8000). Overrides LLM_BACKEND env var.')

    args = parser.parse_args()

    # Override module-level backend globals if --llm-backend is specified
    if args.llm_backend:
        global LLM_BACKEND, LLM_BACKEND_URL
        LLM_BACKEND = args.llm_backend
        LLM_BACKEND_URL = "http://localhost:11435" if LLM_BACKEND == "ollama" else "http://localhost:8000"
    print(f"{GREEN}🔧 LLM backend: {LLM_BACKEND} → {LLM_BACKEND_URL}{RESET}")

    # Create and configure service
    service = RAGLLMAPIService(user_description_server_url=args.user_description_server)
    
    # Auto-initialize if requested
    if args.auto_init:
        print(f"{BLUE}🔄 Auto-initializing RAG system...{RESET}")
        if service._initialize_rag_system():
            print(f"{GREEN}✅ RAG system initialized{RESET}")
        else:
            print(f"{RED}❌ RAG initialization failed{RESET}")
            return 1
    
    # Run the service
    try:
        service.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print(f"\n{BLUE}👋 API service shutting down...{RESET}")
        return 0
    except Exception as e:
        print(f"\n{RED}💥 Service error: {e}{RESET}")
        return 1

if __name__ == "__main__":
    exit(main())