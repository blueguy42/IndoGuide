import json
import os
from typing import List, Dict, Any
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from core.llm_client import LLMClient
from config.config import (
    API_KEY_FILE,
    KNOWLEDGE_BASE_FILE,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    LLM_RERANKER_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    get_prompt
)


class RAGSystem:
    def __init__(self, config: int = 1):
        """
        Initialize RAG system with specified configuration
        
        Args:
            config: Configuration number (1, 2, or 3)
        """
        self.config = config
        
        # Load OpenAI API key
        with open(API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        self.openai_client = OpenAI(api_key=api_key)
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize vector database
        self.collection = self._initialize_vector_db()
        
        # Initialize cross-encoder for config 2
        if self.config == 2:
            print(f"Loading cross-encoder model: {CROSS_ENCODER_MODEL}...")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print("Cross-encoder loaded successfully.")
        
        # Initialize LLM client for config 3
        if self.config == 3:
            self.llm_client = LLMClient(model=LLM_RERANKER_MODEL)
    
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load knowledge base from JSON file"""
        kb_path = os.path.join(os.path.dirname(__file__), "..", KNOWLEDGE_BASE_FILE)
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_vector_db(self) -> chromadb.Collection:
        """
        Initialize or load Chroma vector database
        Embeds only topic, title, and content (not source)
        Stores source as metadata
        """
        # Create Chroma client with persistence
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create collection
        collection_name = "indonesia_knowledge"
        
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            print(f"Loaded existing Chroma collection with {collection.count()} documents.")
        except:
            # Create new collection if it doesn't exist
            print("Creating new Chroma collection...")
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Embed and add all documents
            print(f"Embedding {len(self.knowledge_base)} documents...")
            documents = []
            metadatas = []
            ids = []
            
            for item in self.knowledge_base:
                # Concatenate topic, title, and content for embedding
                text_to_embed = f"{item['topic']} - {item['title']}: {item['content']}"
                documents.append(text_to_embed)
                
                # Store all fields as metadata (including source)
                metadatas.append({
                    "id": str(item["id"]),
                    "topic": item["topic"],
                    "title": item["title"],
                    "content": item["content"],
                    "source": item["source"]
                })
                
                ids.append(str(item["id"]))
            
            # Generate embeddings using OpenAI
            embeddings = self._embed_texts(documents)
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully embedded and stored {len(documents)} documents.")
        
        return collection
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.openai_client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [item.embedding for item in response.data]
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant snippets based on configured method
        
        Args:
            query: User query string
            
        Returns:
            List of top-k snippets with all fields including source
        """
        if self.config == 1:
            return self._retrieve_baseline(query)
        elif self.config == 2:
            return self._retrieve_with_cross_encoder(query)
        elif self.config == 3:
            return self._retrieve_with_llm_reranker(query)
        else:
            raise ValueError(f"Invalid configuration: {self.config}")
    
    def _retrieve_baseline(self, query: str) -> List[Dict[str, Any]]:
        """
        Configuration 1: Baseline retrieval-only
        Retrieve top-10 with vector search, then select top-4 directly
        
        Args:
            query: User query string
            
        Returns:
            Top-4 snippets based on similarity scores
        """
        # Embed query
        query_embedding = self._embed_text(query)
        
        # Retrieve top-10 candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL
        )

        # Extract top-4 from metadata
        snippets = []
        for i in range(min(TOP_K_FINAL, len(results['metadatas'][0]))):
            metadata = results['metadatas'][0][i]
            snippets.append({
                "id": int(metadata["id"]),
                "topic": metadata["topic"],
                "title": metadata["title"],
                "content": metadata["content"],
                "source": metadata["source"]
            })

        return snippets
    
    def _retrieve_with_cross_encoder(self, query: str) -> List[Dict[str, Any]]:
        """
        Configuration 2: Retrieval + Cross-Encoder ReRanker
        Retrieve top-10 with vector search, rerank with cross-encoder, select top-4
        
        Args:
            query: User query string
            
        Returns:
            Top-4 reranked snippets
        """
        # Embed query
        query_embedding = self._embed_text(query)
        
        # Retrieve top-10 candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL
        )

        # Prepare candidates for reranking
        candidates = []
        for i in range(len(results['metadatas'][0])):
            metadata = results['metadatas'][0][i]
            text = f"{metadata['topic']} - {metadata['title']}: {metadata['content']}"
            candidates.append({
                "text": text,
                "metadata": metadata
            })
        
        # Rerank using cross-encoder
        pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by reranker scores
        ranked_candidates = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Extract top-4
        snippets = []
        for i in range(min(TOP_K_FINAL, len(ranked_candidates))):
            metadata = ranked_candidates[i][0]["metadata"]
            snippets.append({
                "id": int(metadata["id"]),
                "topic": metadata["topic"],
                "title": metadata["title"],
                "content": metadata["content"],
                "source": metadata["source"]
            })
        
        return snippets
    
    def _retrieve_with_llm_reranker(self, query: str) -> List[Dict[str, Any]]:
        """
        Configuration 3: Retrieval + LLM-as-ReRanker
        Retrieve top-10 with vector search, rerank with LLM, select top-4
        
        Args:
            query: User query string
            
        Returns:
            Top-4 LLM-reranked snippets
        """
        # Embed query
        query_embedding = self._embed_text(query)
        
        # Retrieve top-10 candidates
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RETRIEVAL
        )

        # Prepare candidates for reranking
        candidates = []
        for i in range(len(results['metadatas'][0])):
            metadata = results['metadatas'][0][i]
            candidates.append({
                "id": int(metadata["id"]),
                "topic": metadata["topic"],
                "title": metadata["title"],
                "content": metadata["content"],
                "source": metadata["source"]
            })
        
        # Create prompt for LLM reranker
        rerank_prompt = self._create_llm_rerank_prompt(query, candidates)
        rerank_system_prompt = get_prompt("llm_reranker_system")
        
        # Call LLM to rerank using Responses API
        response_text = self.llm_client.chat(
            user_message=rerank_prompt,
            system_prompt=rerank_system_prompt,
            auto_add_messages=False  # Don't add to conversation history
        )
        
        # Parse LLM response to get ranked IDs
        ranked_ids = self._parse_llm_rerank_response(response_text)
        
        # Reorder candidates based on LLM ranking
        id_to_candidate = {cand["id"]: cand for cand in candidates}
        snippets = []
        for doc_id in ranked_ids[:TOP_K_FINAL]:
            if doc_id in id_to_candidate:
                snippets.append(id_to_candidate[doc_id])
        
        # If LLM didn't return enough valid IDs, fill with remaining candidates
        if len(snippets) < TOP_K_FINAL:
            for cand in candidates:
                if cand["id"] not in [s["id"] for s in snippets]:
                    snippets.append(cand)
                    if len(snippets) >= TOP_K_FINAL:
                        break
        
        return snippets
    
    def _create_llm_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """
        Create prompt for LLM reranker
        
        Args:
            query: User query
            candidates: List of candidate snippets
            
        Returns:
            Formatted prompt string
        """
        # Format documents for the prompt
        documents_text = ""
        for cand in candidates:
            documents_text += f"ID {cand['id']}: {cand['topic']} - {cand['title']}: {cand['content'][:200]}...\n\n"
        
        # Get prompt template and format it
        prompt_template = get_prompt("llm_reranker_user")
        prompt = prompt_template.format(query=query, documents=documents_text)
        
        return prompt
    
    def _parse_llm_rerank_response(self, response: str) -> List[int]:
        """
        Parse LLM reranker response to extract ranked IDs
        
        Args:
            response: LLM response string
            
        Returns:
            List of document IDs in ranked order
        """
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', response)
        return [int(n) for n in numbers]
    
    def format_context(self, snippets: List[Dict[str, Any]]) -> str:
        """
        Format retrieved snippets for prompt injection
        Includes source for citations
        
        Args:
            snippets: List of retrieved snippets
            
        Returns:
            Formatted context string
        """
        if not snippets:
            return ""
        
        context = "=== RETRIEVED KNOWLEDGE BASE CONTEXT ===\n\n"
        
        for i, snippet in enumerate(snippets, 1):
            context += f"[{i}] Topic: {snippet['topic']}\n"
            context += f"    Title: {snippet['title']}\n"
            context += f"    Content: {snippet['content']}\n"
            context += f"    Source: {snippet['source']}\n\n"
        
        context += "=== END OF CONTEXT ===\n\n"
        context += "Use the above context to answer the user's question.\n"
        
        return context
