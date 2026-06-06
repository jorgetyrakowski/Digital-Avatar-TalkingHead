import os
import shutil
import subprocess
import requests
import chromadb
import json
import re
import argparse
import time
import queue
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import hashlib
import logging
import base64
try:
    import gradio as gr
except Exception:
    gr = None

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        print("Loading embedding model (multilingual-e5-large)...")
        _embed_model = TextEmbedding("intfloat/multilingual-e5-large")
        print("Embedding model ready.")
    return _embed_model

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import LLM_MODEL_NAME, CHROMA_DB_PATH

# Import build_fixed_system_prompt from tone_system_prompts_no_tag
sys.path.insert(0, os.path.join(parent_dir, 'api'))
from tone_system_prompts_no_tag import build_fixed_system_prompt

GRAY = "\033[90m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class ImprovedRAGPipeline:
    def __init__(self, museum_name='itri_museum'):
        self.museum_name = museum_name
        self.chunk_size = 300
        self.chunk_overlap = 50
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.chunk_counter = 0  # Add counter for unique IDs
        self.cached_system_prompt: Optional[str] = None
        
        # Preload jieba once at startup to avoid first-call latency
        self._initialize_jieba()
    
    def _initialize_jieba(self) -> None:
        """Preload jieba dictionary and add domain-specific vocabulary."""
        try:
            custom_terms = [
                "工研院",
                "工業技術研究院",
                "ITRI",
                "人才培育",
                "產業升級",
                "創新技術",
                "中興院區",
                "光復院區",
                "南分院",
                "六甲院區"
            ]
            for term in custom_terms:
                # Some jieba versions require freq as a positional argument
                try:
                    jieba.add_word(term, freq=1)
                except TypeError:
                    # Fallback for older/newer signatures
                    jieba.add_word(term)

            sample_texts = [
                "工研院是台灣最重要的產業技術研發機構",
                "ITRI 推動產業升級與人才培育",
            ]
            for sample_text in sample_texts:
                _ = list(jieba.cut(sample_text))

            print(f"{GREEN}✅ Jieba initialized and preloaded{RESET}")
        except Exception as error:
            print(f"{YELLOW}⚠️ Jieba preload failed: {error}{RESET}")

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for Chinese and English"""
        # Remove special characters but keep Chinese characters
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use jieba for Chinese word segmentation
        words = jieba.cut(text)
        return ' '.join(words)
    
    def load_json_data(self, data_dir: str = "itri_museum_docs") -> List[DocumentChunk]:
        """Load and process JSON data from the Wikipedia crawler - traverses all subfolders"""
        all_chunks = []
        
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            return all_chunks
        
        # Recursively find all JSON files in the directory and subdirectories
        json_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        print(f"Found {len(json_files)} JSON files in {data_dir} and subdirectories:")
        
        # Process each JSON file
        for json_file in json_files:
            file_name = os.path.basename(json_file)
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Prefer content-aware handling so golden JSON lists are parsed cleanly
                if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("content"):
                    chunks = self._process_itri_golden_entries(data, file_name)
                elif 'raw_data' in file_name:
                    chunks = self._process_raw_data(data, file_name)
                elif 'qa_pairs' in file_name:
                    chunks = self._process_qa_pairs(data, file_name)
                elif 'structured_data' in file_name:
                    chunks = self._process_structured_data(data, file_name)
                else:
                    # Generic processing for unknown JSON files
                    chunks = self._process_generic_json(data, file_name)
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        return all_chunks
    
    def _process_raw_data(self, data: Dict, file_name: str) -> List[DocumentChunk]:
        """Process raw data JSON files"""
        chunks = []
        
        # Process organization info
        if 'organization_name' in data:
            org_text = f"組織名稱: {data['organization_name']}"
            chunks.append(DocumentChunk(
                content=org_text,
                chunk_id=f"org_info_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'organization_info',
                    'length': len(org_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', org_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        # Process sections
        if 'sections' in data:
            for section_name, content in data['sections'].items():
                if isinstance(content, dict):
                    content_str = json.dumps(content, ensure_ascii=False, indent=2)
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = str(content)
                
                if content_str and len(content_str.strip()) > 0:
                    section_chunks = self.semantic_chunking(content_str, f"section_{section_name}")
                    for i, chunk in enumerate(section_chunks):
                        chunk.chunk_id = f"section_{section_name}_{self.chunk_counter}"
                        chunk.source_file = file_name
                        chunk.chunk_index = self.chunk_counter
                        chunk.metadata.update({
                            'section_name': section_name,
                            'type': 'section'
                        })
                        self.chunk_counter += 1
                    chunks.extend(section_chunks)
        
        # Process achievements
        if 'achievements' in data and data['achievements']:
            achievements_text = "重要成就: " + "；".join(data['achievements'])
            chunks.append(DocumentChunk(
                content=achievements_text,
                chunk_id=f"achievements_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'achievements',
                    'length': len(achievements_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', achievements_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        # Process leadership
        if 'leadership' in data:
            for position, leaders in data['leadership'].items():
                if leaders:
                    leadership_text = f"{position}: {', '.join(leaders)}"
                    chunks.append(DocumentChunk(
                        content=leadership_text,
                        chunk_id=f"leadership_{position}_{self.chunk_counter}",
                        source_file=file_name,
                        chunk_index=self.chunk_counter,
                        metadata={
                            'type': 'leadership',
                            'position': position,
                            'length': len(leadership_text),
                            'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', leadership_text) else 'english'
                        }
                    ))
                    self.chunk_counter += 1
        
        return chunks
    
    def _process_itri_golden_entries(self, data: List[Dict], file_name: str) -> List[DocumentChunk]:
        """Process golden JSON list where each item contains content/title/source/hierarchy."""
        chunks: List[DocumentChunk] = []
        for entry in data:
            content = entry.get("content", "")
            if not isinstance(entry, dict) or not content or not str(content).strip():
                continue
            
            title = entry.get("title", "Untitled")
            hierarchy = entry.get("hierarchy", "")
            source = entry.get("source", "")
            category = entry.get("category", "")
            unit_name = entry.get("unit_name", "")
            leader = entry.get("leader", "")
            position = entry.get("position", "")
            year = entry.get("year")

            base_metadata = {
                "type": "itri_golden",
                "title": title,
                "hierarchy": hierarchy,
                "source": source,
                "category": category,
                "unit_name": unit_name,
                "leader": leader,
                "position": position,
                "year": year,
            }

            golden_chunks = self.semantic_chunking(content, f"golden_{file_name}")
            for chunk in golden_chunks:
                chunk.chunk_id = f"golden_{self.chunk_counter}"
                chunk.source_file = file_name
                chunk.chunk_index = self.chunk_counter
                chunk.metadata.update(base_metadata)
                self.chunk_counter += 1
            chunks.extend(golden_chunks)
        
        return chunks
    
    def _process_qa_pairs(self, data: List[Dict], file_name: str) -> List[DocumentChunk]:
        """Process Q&A pairs JSON files"""
        chunks = []
        for i, qa in enumerate(data):
            qa_text = f"問題: {qa.get('question', '')} 答案: {qa.get('answer', '')}"
            chunks.append(DocumentChunk(
                content=qa_text,
                chunk_id=f"qa_pair_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'qa_pair',
                    'question': qa.get('question', ''),
                    'answer': qa.get('answer', ''),
                    'length': len(qa_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', qa_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        return chunks
    
    def _process_structured_data(self, data: Dict, file_name: str) -> List[DocumentChunk]:
        """Process structured data JSON files"""
        chunks = []
        
        if 'organization_info' in data:
            org_info = data['organization_info']
            org_text = f"組織名稱: {org_info.get('name', '')} 描述: {org_info.get('description', '')}"
            chunks.append(DocumentChunk(
                content=org_text,
                chunk_id=f"structured_org_info_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'structured_organization_info',
                    'length': len(org_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', org_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        if 'key_facts' in data:
            key_facts = data['key_facts']
            facts_text = " ".join([f"{k}: {v}" for k, v in key_facts.items()])
            chunks.append(DocumentChunk(
                content=facts_text,
                chunk_id=f"key_facts_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'key_facts',
                    'length': len(facts_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', facts_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        if 'sections' in data:
            for section_name, content in data['sections'].items():
                if isinstance(content, dict):
                    content_str = json.dumps(content, ensure_ascii=False, indent=2)
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = str(content)
                
                if content_str and len(content_str.strip()) > 0:
                    section_chunks = self.semantic_chunking(content_str, f"structured_section_{section_name}")
                    for i, chunk in enumerate(section_chunks):
                        chunk.chunk_id = f"structured_section_{section_name}_{self.chunk_counter}"
                        chunk.source_file = file_name
                        chunk.chunk_index = self.chunk_counter
                        chunk.metadata.update({
                            'section_name': section_name,
                            'type': 'structured_section'
                        })
                        self.chunk_counter += 1
                    chunks.extend(section_chunks)
        
        if 'achievements' in data and data['achievements']:
            achievements_text = "重要成就: " + "；".join(data['achievements'])
            chunks.append(DocumentChunk(
                content=achievements_text,
                chunk_id=f"structured_achievements_{self.chunk_counter}",
                source_file=file_name,
                chunk_index=self.chunk_counter,
                metadata={
                    'type': 'structured_achievements',
                    'length': len(achievements_text),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', achievements_text) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        if 'leadership' in data:
            for position, leaders in data['leadership'].items():
                if leaders:
                    leadership_text = f"{position}: {', '.join(leaders)}"
                    chunks.append(DocumentChunk(
                        content=leadership_text,
                        chunk_id=f"structured_leadership_{position}_{self.chunk_counter}",
                        source_file=file_name,
                        chunk_index=self.chunk_counter,
                        metadata={
                            'type': 'structured_leadership',
                            'position': position,
                            'length': len(leadership_text),
                            'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', leadership_text) else 'english'
                        }
                    ))
                    self.chunk_counter += 1
        
        return chunks
    
    def _process_generic_json(self, data: Any, file_name: str) -> List[DocumentChunk]:
        """Process generic JSON files by converting to text"""
        chunks = []
        
        def extract_text_from_json(obj, prefix=""):
            if isinstance(obj, dict):
                text_parts = []
                for key, value in obj.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, (list, dict)):
                        text_parts.append(f"{key}: {extract_text_from_json(value, f'{prefix}{key}_')}")
                return " ".join(text_parts)
            elif isinstance(obj, list):
                text_parts = []
                for i, item in enumerate(obj):
                    if isinstance(item, (str, int, float)):
                        text_parts.append(str(item))
                    elif isinstance(item, (dict, list)):
                        text_parts.append(extract_text_from_json(item, f'{prefix}{i}_'))
                return " ".join(text_parts)
            else:
                return str(obj)
        
        text_content = extract_text_from_json(data)
        
        if text_content.strip():
            text_chunks = self.semantic_chunking(text_content, f"generic_{file_name}")
            for i, chunk in enumerate(text_chunks):
                chunk.chunk_id = f"generic_{file_name}_{self.chunk_counter}"
                chunk.source_file = file_name
                chunk.chunk_index = self.chunk_counter
                chunk.metadata.update({
                    'type': 'generic_json',
                    'length': len(chunk.content),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', chunk.content) else 'english'
                })
                self.chunk_counter += 1
            chunks.extend(text_chunks)
        
        return chunks
    
    def semantic_chunking(self, text: str, source_file: str) -> List[DocumentChunk]:
        """Advanced semantic chunking that respects sentence boundaries"""
        chunks = []
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunk_id = f"{source_file}_{self.chunk_counter}"
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=chunk_id,
                        source_file=source_file,
                        chunk_index=self.chunk_counter,
                        metadata={
                            'length': len(current_chunk),
                            'sentence_count': current_chunk.count('。') + current_chunk.count('!') + current_chunk.count('?'),
                            'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', current_chunk) else 'english'
                        }
                    ))
                    self.chunk_counter += 1
                    
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunk_id = f"{source_file}_{self.chunk_counter}"
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                source_file=source_file,
                chunk_index=self.chunk_counter,
                metadata={
                    'length': len(current_chunk),
                    'sentence_count': current_chunk.count('。') + current_chunk.count('!') + current_chunk.count('?'),
                    'language': 'chinese' if re.search(r'[\u4e00-\u9fff]', current_chunk) else 'english'
                }
            ))
            self.chunk_counter += 1
        
        return chunks

    def load_and_chunk_docs(self, folder: str) -> List[DocumentChunk]:
        """Enhanced document loading with semantic chunking and JSON support"""
        all_chunks = []
        json_chunks = self.load_json_data(folder)
        all_chunks.extend(json_chunks)
        
        if os.path.exists(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                for fname in filenames:
                    if fname.endswith('.txt'):
                        file_path = os.path.join(dirpath, fname)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            chunks = self.semantic_chunking(text, fname)
                            all_chunks.extend(chunks)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
                            continue
        
        print(f"Created {len(all_chunks)} total chunks from JSON and text documents")
        return all_chunks

    def build_hybrid_vector_store(self, chunks: List[DocumentChunk], collection_name: str, 
                                 embedding_model: str = "bge-m3:latest", reload: bool = False):
        """Build hybrid vector store with both dense and sparse embeddings"""
        
        chroma_db_path = CHROMA_DB_PATH
        os.makedirs(chroma_db_path, exist_ok=True)
        os.chmod(chroma_db_path, 0o755)
        
        if reload:
            print(f"Reloading vector store - cleaning up old data...")
            import shutil
            if os.path.exists(chroma_db_path):
                try:
                    shutil.rmtree(chroma_db_path)
                    print(f"Cleared ChromaDB directory: {chroma_db_path}")
                    os.makedirs(chroma_db_path, exist_ok=True)
                    os.chmod(chroma_db_path, 0o755)
                except Exception as e:
                    print(f"Warning: Could not clear ChromaDB directory: {e}")
            
            self.vectorizer = None
            self.tfidf_matrix = None
            self.chunks = []
            
            chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            try:
                chroma_collection = chroma_client.create_collection(collection_name)
                print(f"Created new ChromaDB collection: {collection_name}")
            except Exception as e:
                print(f"Error creating collection: {e}")
                try:
                    chroma_client.delete_collection(collection_name)
                    chroma_collection = chroma_client.create_collection(collection_name)
                    print(f"Successfully recreated ChromaDB collection: {collection_name}")
                except Exception as e2:
                    print(f"Failed to recreate collection: {e2}")
                    raise
            
            self._build_tfidf_index(chunks)
            
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            print(f"Generating embeddings for {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:
                    print(f"\rProcessing chunk {i+1}/{len(chunks)}", end="", flush=True)
                
                embedding = list(_get_embed_model().embed([f"passage: {chunk.content}"]))[0].tolist()
                
                embeddings.append(embedding)
                documents.append(chunk.content)
                
                # Ensure metadata values are strings or primitives
                safe_metadata = {}
                for k, v in chunk.metadata.items():
                    if v is None: continue
                    safe_metadata[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
                
                metadatas.append(safe_metadata)
                ids.append(chunk.chunk_id)
            
            print()
            
            BATCH_SIZE = 5000
            print(f"Storing {len(documents)} chunks in ChromaDB...")
            for i in range(0, len(documents), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(documents))
                chroma_collection.add(
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
                print(f"Stored batch {i//BATCH_SIZE + 1}/{(len(documents) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            print(f"Successfully stored {len(documents)} document chunks in hybrid vector store")
            
        else:
            chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            try:
                chroma_collection = chroma_client.get_collection(collection_name)
                print(f"✅ Loaded existing ChromaDB collection '{collection_name}'")
                
                if not self.vectorizer or self.tfidf_matrix is None:
                    print(f"🔧 Building TF-IDF index for sparse retrieval...")
                    self._build_tfidf_index(chunks)
                
                return chroma_collection, embedding_model
            except Exception as e:
                print(f"❌ Failed to load existing collection: {e}")
                raise Exception(f"Cannot load existing collection '{collection_name}': {e}")
        
        return chroma_collection, embedding_model

    def _build_tfidf_index(self, chunks: List[DocumentChunk]):
        """Build TF-IDF index for sparse retrieval"""
        documents = [chunk.content for chunk in chunks]
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.chunks = chunks
        print(f"Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")

    def hybrid_search(self, query: str, chroma_collection, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval"""
        
        # Dense search (ChromaDB)
        try:
            q_emb = [list(_get_embed_model().embed([f"query: {query}"]))[0].tolist()]
            
            dense_results = chroma_collection.query(
                query_embeddings=q_emb, 
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"Dense search failed: {e}")
            dense_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        # Sparse search (TF-IDF)
        sparse_results = []
        if self.vectorizer and self.tfidf_matrix is not None:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                for idx in top_indices:
                    if similarities[idx] > 0:
                        sparse_results.append({
                            'content': self.chunks[idx].content,
                            'metadata': self.chunks[idx].metadata,
                            'score': float(similarities[idx]),
                            'chunk_id': self.chunks[idx].chunk_id
                        })
            except Exception as e:
                print(f"Sparse search failed: {e}")

        combined_results = self._combine_and_rerank(dense_results, sparse_results, top_k)
        return combined_results

    def _combine_and_rerank(self, dense_results: Dict, sparse_results: List[Dict], top_k: int) -> List[Dict[str, Any]]:
        """Combine dense and sparse results with intelligent reranking"""
        
        content_scores = {}
        
        if dense_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                dense_results['documents'][0], 
                dense_results['metadatas'][0], 
                dense_results['distances'][0]
            )):
                similarity = 1.0 / (1.0 + distance)
                content_scores[doc] = {
                    'content': doc,
                    'metadata': metadata,
                    'dense_score': similarity,
                    'sparse_score': 0.0,
                    'combined_score': similarity
                }
        
        for result in sparse_results:
            content = result['content']
            if content in content_scores:
                content_scores[content]['sparse_score'] = result['score']
                content_scores[content]['combined_score'] = (
                    0.7 * content_scores[content]['dense_score'] + 
                    0.3 * result['score']
                )
            else:
                content_scores[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'sparse_score': result['score'],
                    'combined_score': result['score']
                }
        
        sorted_results = sorted(
            content_scores.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        return sorted_results[:top_k]

    def _process_context(self, context_chunks: List[str], query: str) -> str:
        """
        Enhanced context processing.
        CRITICAL FIX: Removed sort-by-length logic which was discarding detailed (long) chunks.
        """
        if not context_chunks:
            return ""
        
        processed_chunks = []
        current_length = 0
        MAX_LENGTH = 2500  # Increased slightly for better context
        
        # Directly use the sorted chunks from hybrid search (most relevant first)
        for chunk in context_chunks:
            if not chunk or not chunk.strip():
                continue
                
            chunk_len = len(chunk)
            
            if current_length + chunk_len > MAX_LENGTH:
                # If we haven't added anything yet, add at least one truncated chunk
                if not processed_chunks:
                    processed_chunks.append(chunk[:MAX_LENGTH])
                break
            
            processed_chunks.append(chunk)
            current_length += chunk_len
        
        return "\n\n".join(processed_chunks)


    def _build_messages(self, system_prompt: str, user_question: str,
                        history: Optional[List[Dict]], rag_reference: Optional[str], 
                        user_description: Optional[str] = None, rewritten_query: Optional[str] = None) -> List[Dict]:
        """Build messages where the user content is a JSON payload with user_question, chat_history, rag_reference, optional user_description, and optional rewritten_query."""

        messages = [{"role": "system", "content": system_prompt}]

        structured_history: List[Dict[str, Any]] = []
        question_count = 0
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    question_count += 1
                    structured_history.append({
                        "id": f"Q{question_count}",
                        "role": "user",
                        "content": msg.get("content", "")
                    })
                elif msg.get("role") == "assistant":
                    structured_history.append({
                        "id": f"A{question_count}",
                        "role": "assistant",
                        "content": msg.get("content", "")
                    })

        # Detect user question language
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in user_question)
        detected_language = "Traditional Chinese (繁體中文)" if has_chinese else "English"
        
        user_payload = {
            "user_question": user_question,
            "chat_history": structured_history,
            "rag_reference": rag_reference or "",
            "language_requirement": f"CRITICAL: The user question is in {detected_language}. You MUST respond ENTIRELY in {detected_language}. Do NOT use any other language."
        }
        
        # Add user_description if provided (for vision context awareness)
        if user_description and user_description.strip():
            user_payload["user_description"] = user_description.strip()
        
        # Add rewritten_query if provided (for better understanding of user intent)
        if rewritten_query and rewritten_query.strip() and rewritten_query != user_question:
            user_payload["rewritten_query"] = rewritten_query.strip()
            print(f"{BLUE}📝 Rewritten query included in QA context: '{rewritten_query}'{RESET}")

        messages.append({
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False)
        })
        return messages

    def _probe_duration_seconds(self, p: str) -> float:
        try:
            if shutil.which("ffprobe"):
                proc = subprocess.run(
                    ["ffprobe","-v","error","-show_entries","format=duration","-of","json",p],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True
                )
                data = json.loads(proc.stdout)
                dur = float(data.get("format", {}).get("duration", 0.0))
                if dur > 0:
                    return dur
        except Exception:
            pass
        try:
            from mutagen.mp3 import MP3
            return float(MP3(p).info.length)
        except Exception:
            pass
        return 0.0

    def _handle_new_segment(self, path: str,
                            first_played: bool,
                            current_end_ts: Optional[float],
                            queue: List[tuple],
                            extra_first_buffer: float,
                            queued_delta: float):
        immediate_paths: List[tuple] = []  # list of (path, dur, is_first_here)
        dur = self._probe_duration_seconds(path) or 0.0
        if not first_played:
            immediate_paths.append((path, dur, True))
            first_played = True
            current_end_ts = time.time() + max(dur + extra_first_buffer, 0.1)
        else:
            queue.append((path, max(dur + queued_delta, 0.05)))
            while queue and current_end_ts and time.time() >= current_end_ts:
                nxt, ndur = queue.pop(0)
                immediate_paths.append((nxt, ndur, False))
                current_end_ts = time.time() + max(ndur, 0.1)
        return first_played, current_end_ts, immediate_paths, queue
        
def main():
    parser = argparse.ArgumentParser(description='RAG Pipeline with reload control')
    parser.add_argument('--RAG_RELOAD', action='store_true', default=False,
                       help='Reload the vector database (default: False, use existing if available)')
    parser.add_argument('--gradio', action='store_true', default=False,
                       help='Launch a Gradio UI that autoplays TTS for Enhanced RAG')
    args = parser.parse_args()
    
    rag_pipeline = ImprovedRAGPipeline(museum_name='itri_museum')
    rag_pipeline.cached_system_prompt = build_fixed_system_prompt(
        "NOTICE: The response language should depend on what the user requires. If the user does not specify a language, use the same language as the user input. If Mandarin or Chinese is requested, respond in Traditional Chinese (zh-tw). Most importantly, ensure all information comes from rag_reference and is accurate and truthful."
    )

    try:
        warmup_messages = [
            {"role": "system", "content": rag_pipeline.cached_system_prompt},
            {"role": "user", "content": json.dumps({
                "user_question": "Warm up and acknowledge readiness.",
                "chat_history": [],
                "rag_reference": ""
            }, ensure_ascii=False)}
        ]
        warmup_payload = {
            "model": f"{LLM_MODEL_NAME}",
            "messages": warmup_messages,
            "stream": True,
            "options": {"temperature": 0}
        }
        _resp = requests.post("http://localhost:11435/api/chat", json=warmup_payload, stream=False)
        if _resp.ok:
            for _line in _resp.iter_lines():
                if not _line:
                    continue
                try:
                    _chunk = json.loads(_line)
                except Exception:
                    continue
                if _chunk.get("done"):
                    break
        print("Model warm-up request sent.")
    except Exception as _e:
        print(f"Warm-up skipped: {_e}")
    
    _llm_chat_dir = os.path.dirname(os.path.abspath(__file__))
    docs_folder = os.path.join(_llm_chat_dir, '..', 'knowledge')
    chunks = rag_pipeline.load_and_chunk_docs(docs_folder)
    
    chroma_collection, embedding_model = rag_pipeline.build_hybrid_vector_store(
        chunks, collection_name=f"{rag_pipeline.museum_name}_collection",
        reload=args.RAG_RELOAD
    )
    time.sleep(3)
    
    test_queries = [
        "工研院有哪些重要成就？",
        "工研院的組織架構如何？",
        "工研院的院長是誰？",
    ]

    def _build_gradio_app(rag_pipeline: ImprovedRAGPipeline, chroma_collection, embedding_model: str):
        if gr is None:
            raise RuntimeError("gradio is not installed. Please install gradio to use the UI.")

        def _ask_stream(user_question: str, clear_history: bool, history: Optional[List[Dict[str, str]]] = None, _unused_queue: Optional[List[str]] = None):
            history = history or []
            pending_audio: List[tuple] = []
            current_play_end_ts: Optional[float] = None
            if clear_history:
                history = []

            try:
                search_results = rag_pipeline.hybrid_search(user_question, chroma_collection, top_k=10)
            except Exception:
                search_results = []

            museum_context = []
            for result in search_results:
                content = result.get('content', '')
                if content and not (content.startswith('[Q') or content.startswith('[A')):
                    museum_context.append(content)
            context = rag_pipeline._process_context(museum_context, user_question)

            if not rag_pipeline.cached_system_prompt:
                rag_pipeline.cached_system_prompt = build_fixed_system_prompt(
                    "NOTICE: The response language should depend on what the user requires. If the user does not specify a language, use the same language as the user input. If Mandarin or Chinese is requested, respond in Traditional Chinese (zh-tw). Most importantly, ensure all information comes from rag_reference and is accurate and truthful."
                )
            system_prompt = rag_pipeline.cached_system_prompt
            messages = rag_pipeline._build_messages(system_prompt, user_question, history, context)

            try:
                from edge_tts_helper import EdgeTTSClient
                tts_client = EdgeTTSClient(voice="en-US-BrianMultilingualNeural")
            except Exception:
                tts_client = None

            import requests as _req
            response = _req.post(
                "http://localhost:11435/api/generate",
                json={
                    "model": f"{LLM_MODEL_NAME}",
                    "prompt": "".join(
                        [f"{m['content']}\n\n" if m['role']=="system" else f"User: {m['content']}\nAssistant: " for m in messages]
                    ),
                    "stream": True,
                    "num_ctx": 32768,
                    "flash_attn": 1,
                },
                stream=True,
            )
            response.raise_for_status()

            answer_so_far = ""
            sentence_buf = ""
            first_audio_played = False

            pretty = []
            for turn in history[-6:]:
                pretty.append(f"{turn.get('role', '')}: {turn.get('content', '')}")
            transcript_val = "\n".join(pretty)
            yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(value=None), gr.update(value=[]), history

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                delta = chunk.get("response")
                if delta:
                    answer_so_far += delta
                    sentence_buf += delta

                    yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(), gr.update(value=[p for p,_ in pending_audio]), history

                    parts = re.split(r"[，,。、.!?！？：；\n]+", sentence_buf)
                    if len(parts) > 1:
                        complete_segments = parts[:-1]
                        sentence_buf = parts[-1]
                        print(f"{YELLOW}complete_segments:{RESET} {complete_segments}")
                        for seg in complete_segments:
                            seg = seg.strip()
                            if not seg:
                                continue
                            try:
                                if tts_client and tts_client.is_available():
                                    import hashlib as _hl
                                    name = _hl.md5(f"{seg}-{time.time()}".encode("utf-8")).hexdigest()[:16]
                                    path = tts_client.synth_to_file(seg, rate=0, volume=85, pitch_hz=0, file_basename=name)
                                    if path and os.path.exists(path):
                                        first_audio_played, current_play_end_ts, now_list, pending_audio = rag_pipeline._handle_new_segment(
                                            path, first_audio_played, current_play_end_ts, pending_audio, extra_first_buffer=0, queued_delta=-0.03
                                        )
                                        yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(value=now_list[0][0]), gr.update(value=[p for p,_ in pending_audio]), history
                                        pre_play_dur = now_list[0][1]
                                        print(f"{GREEN}show:{RESET} {now_list[0][0]}")
                                        if len(now_list) > 1:
                                            for play_path, play_dur, is_first_here in now_list[1:]:
                                                print(f"{GREEN}show:{RESET} {play_path}")
                                                yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(value=play_path), gr.update(value=[p for p,_ in pending_audio]), history
                                                pre_play_dur = play_dur
                                        else:
                                            time.sleep(max(pre_play_dur, 0.01))
                            except Exception:
                                pass
                elif chunk.get("done"):
                    print(f"{BLUE}show:{RESET} {chunk.get('done')}")
                    break

            history = history.copy()
            history.append({"role": "user", "content": user_question})
            history.append({"role": "assistant", "content": answer_so_far})
            while pending_audio:
                nxt, ndur = pending_audio.pop(0)
                print(f"{RED}show:{RESET} {nxt}")
                yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(value=nxt), gr.update(value=[p for p,_ in pending_audio]), history
                time.sleep(max(ndur, 0.01))

            yield gr.update(value=answer_so_far), gr.update(value=transcript_val), gr.update(), gr.update(value=[]), history

        with gr.Blocks(title="ITRI Museum RAG + TTS") as demo:
            history_state = gr.State([])
            audio_queue_state = gr.State([])
            gr.Markdown("**ITRI Museum RAG Demo** — Streaming TTS per sentence; audio should play as segments are ready.")
            with gr.Row():
                q = gr.Textbox(label="Question", placeholder="輸入你的問題 / Ask your question")
            with gr.Row():
                clear = gr.Checkbox(label="Clear chat history before asking", value=False)
                ask_btn = gr.Button("Ask")
            with gr.Row():
                ans = gr.Textbox(label="Answer", lines=2)
            with gr.Row():
                transcript = gr.Textbox(label="Recent conversation (last 3 turns)", lines=6)
            audio = gr.Audio(label="TTS", autoplay=True, type="filepath", elem_id="tts_player")

            ask_btn.click(
                _ask_stream,
                inputs=[q, clear, history_state, audio_queue_state],
                outputs=[ans, transcript, audio, audio_queue_state, history_state],
            )

        return demo

    def _test_original_query(rag_pipeline, chroma_collection, embedding_model, test_queries):
        _, _ = test_enhanced_rag_query(rag_pipeline, chroma_collection, embedding_model, test_queries)

    def test_enhanced_rag_query(rag_pipeline, chroma_collection, embedding_model, test_queries):
        print("=" * 80)
        print("TESTING ENHANCED RAG QUERY METHOD (HYBRID)")
        print("=" * 80)
        
        enhanced_rag_chat_history = []
        
        for i, query in enumerate(test_queries):
            print(f"\nQuestion {i+1}: {query}")
            
            search_results = rag_pipeline.hybrid_search(query, chroma_collection, top_k=10)
            
            museum_context = []
            conversation_context = []
            
            for result in search_results:
                content = result['content']
                if content.startswith('[Q') or content.startswith('[A'):
                    conversation_context.append(content)
                else:
                    museum_context.append(content)
            
            context = rag_pipeline._process_context(museum_context, query)
            
            system_prompt = build_fixed_system_prompt(
                "NOTICE: The response language should depend on what the user requires. If the user does not specify a language, use the same language as the user input. If Mandarin or Chinese is requested, respond in Traditional Chinese (zh-tw). Most importantly, ensure all information comes from rag_reference and is accurate and truthful."
            )
            
            messages = rag_pipeline._build_messages(system_prompt, query, enhanced_rag_chat_history, context)
            
            request_payload = {
                "model": f"{LLM_MODEL_NAME}",
                "messages": messages,
                "stream": True
            }
            
            start_ts = time.time()
            first_llm_token_logged = False
            first_tts_logged = False
            response = requests.post("http://localhost:11435/api/chat", json=request_payload, stream=True)
            
            response.raise_for_status()
            
            response_content = ""

            tts_queue: "queue.Queue[str]" = queue.Queue()
            tts_thread: Optional[threading.Thread] = None
            edge_tts_client = None
            try:
                from edge_tts_helper import EdgeTTSClient
                edge_tts_client = EdgeTTSClient(voice="en-US-BrianMultilingualNeural")
            except Exception as e:
                print(f"Error initializing EdgeTTSClient: {e}")
                edge_tts_client = None

            def _tts_worker():
                nonlocal first_tts_logged
                if not edge_tts_client or not edge_tts_client.is_available():
                    return
                while True:
                    sentence = tts_queue.get()
                    if sentence is None or sentence == "":
                        tts_queue.task_done()
                        break
                    try:
                        basename = hashlib.md5(f"{sentence}-{time.time()}".encode("utf-8")).hexdigest()[:16]
                        out_path = os.path.join(edge_tts_client.out_dir, f"{basename}.mp3")
                        edge_tts_client.synth_to_file(
                            sentence, rate=0, volume=85, pitch_hz=0, file_basename=basename
                        )
                        print(f"[TTS] {out_path}")
                        if not first_tts_logged:
                            print(f"{RED}[METRIC] first_tts_elapsed_s={time.time() - start_ts:.3f}{RESET}")
                            first_tts_logged = True
                        player = None
                        for cand in ("ffplay", "mpv", "mpg123"):
                            if shutil.which(cand):
                                player = cand
                                break
                        try:
                            if player == "ffplay":
                                subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            elif player == "mpv":
                                subprocess.Popen(["mpv", "--no-video", "--really-quiet", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            elif player == "mpg123":
                                subprocess.Popen(["mpg123", "-q", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            else:
                                print("[TTS] No audio player found (tried ffplay/mpv/mpg123).")
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Error synthesizing to file: {e}")
                        pass
                    tts_queue.task_done()

            if edge_tts_client and edge_tts_client.is_available():
                tts_thread = threading.Thread(target=_tts_worker, daemon=True)
                tts_thread.start()
            else:
                print("[TTS] Edge TTS unavailable; skipping.")

            sentence_buf = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                message = chunk.get("message", {})
                delta = message.get("content")
                if delta:
                    print(delta, end="", flush=True)
                    response_content += delta
                    if not first_llm_token_logged:
                        print(f"\n{RED}[METRIC] first_llm_token_elapsed_s={time.time() - start_ts:.3f}{RESET}")
                        first_llm_token_logged = True
                    sentence_buf += delta
                    parts = re.split(r"[，,。、.!?！？：；\n]+", sentence_buf)
                    if len(parts) > 1:
                        complete = parts[:-1]
                        remainder = parts[-1]
                        for s in complete:
                            s = s.strip()
                            if not s:
                                continue
                            try:
                                tts_queue.put_nowait(s)
                            except Exception:
                                pass
                        sentence_buf = remainder
                if chunk.get("done"):
                    break
            print()
            try:
                if sentence_buf.strip():
                    tts_queue.put_nowait(sentence_buf.strip())
                tts_queue.put_nowait(None)
            except Exception as e:
                print(f"Error putting nowait: {e}")
                pass
            try:
                if tts_thread and tts_thread.is_alive():
                    tts_thread.join(timeout=2.0)
            except Exception as e:
                print(f"Error joining thread: {e}")
                pass
            
            enhanced_rag_chat_history.append({"role": "user", "content": query})
            enhanced_rag_chat_history.append({"role": "assistant", "content": response_content})
            
            print(f"Enhanced RAG Answer: {response_content}")
            print("-" * 50)
            time.sleep(1)
        
        return [], []
    
    if args.gradio:
        try:
            app = _build_gradio_app(rag_pipeline, chroma_collection, embedding_model)
            app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
        except Exception as _e:
            print(f"Failed to launch Gradio UI: {_e}")
        return
    else:
        try:
            _test_original_query(rag_pipeline, chroma_collection, embedding_model, test_queries)
        except Exception as _e:
            print(f"Post-build test skipped (LLM not available): {_e}")


if __name__ == "__main__":
    main()

