import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Placeholder for a real vector DB library
# For now, we'll simulate it with a list and basic search
# In a real scenario, you'd import and use a library like FAISS, Annoy, etc.

class MemoryDBHandler:
    _instance = None
    _model_name = 'all-MiniLM-L6-v2'
    _embedding_dim = 384 # Dimension for 'all-MiniLM-L6-v2'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryDBHandler, cls).__new__(cls)
            cls._instance._initialize_db()
        return cls._instance

    def _initialize_db(self):
        """Initializes the in-memory vector database and sentence transformer model."""
        self.sentence_model = SentenceTransformer(self._model_name)
        # Using IndexFlatL2 for simplicity, can be changed for larger datasets
        # IndexIDMap allows us to use our own IDs for vectors
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self._embedding_dim))
        self.memories_data: Dict[str, Dict[str, Any]] = {}  # Stores memory_id -> {text, timestamp, tags, url}
        self.memory_id_to_faiss_id: Dict[str, int] = {}  # Maps our memory_id to FAISS internal ID
        self.next_faiss_id: int = 0 # Simple incremental ID for FAISS

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for the given text."""
        embedding = self.sentence_model.encode([text], convert_to_numpy=True)
        return embedding.astype('float32') # FAISS expects float32

    def add_memory(self, text: str, tags: Optional[List[str]] = None, url: Optional[str] = None, embedding: Optional[List[float]] = None) -> str:
        """
        Adds a text memory to the database.
        Generates an embedding if not provided (embedding arg is mostly for compatibility, will be ignored).
        Accepts an optional list of tags and an optional URL.
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        actual_embedding = self._generate_embedding(text)
        
        faiss_id = self.next_faiss_id
        self.next_faiss_id += 1

        self.index.add_with_ids(actual_embedding, np.array([faiss_id]))
        
        self.memories_data[memory_id] = {
            "text": text,
            "timestamp": timestamp,
            "tags": tags if tags is not None else [],
            "url": url
            # Store string representation or skip if too large and re-embed on query if needed
            # For now, not storing embedding here to avoid redundancy with FAISS index
        }
        self.memory_id_to_faiss_id[memory_id] = faiss_id

        return memory_id

    def remove_memory(self, memory_id: str) -> bool:
        """Removes a chunk by its ID from both data store and FAISS index."""
        if memory_id not in self.memories_data:
            return False

        faiss_id_to_remove = self.memory_id_to_faiss_id.get(memory_id)
        if faiss_id_to_remove is not None:
            try:
                # FAISS remove_ids returns the number of elements removed
                num_removed = self.index.remove_ids(np.array([faiss_id_to_remove]))
                if num_removed == 0:
                    # This might happen if the ID wasn't in the index for some reason
                    pass # Or log a warning
            except RuntimeError as e:
                # Some FAISS index types might not support remove_ids directly or efficiently
                # For IndexFlatL2, it should work. Handle error if it occurs.
                print(f"Error removing from FAISS index: {e}") # TODO: Proper logging
                # In a production system, might need to rebuild index or use a different strategy
                pass 

        del self.memories_data[memory_id]
        if faiss_id_to_remove is not None:
            del self.memory_id_to_faiss_id[memory_id]
        
        return True

    def query_memories_by_recency(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Queries memories sorted by recency (newest first)."""
        sorted_memory_ids = sorted(
            self.memories_data.keys(),
            key=lambda mid: self.memories_data[mid]["timestamp"],
            reverse=True
        )
        results = []
        for mid in sorted_memory_ids[:limit]:
            data = self.memories_data[mid]
            results.append({
                "id": mid,
                "text": data["text"],
                "timestamp": data["timestamp"],
                "tags": data.get("tags", []),
                "url": data.get("url")
            })
        return results

    def query_memories_by_similarity(
        self,
        query_text: str, 
        query_embedding: Optional[List[float]] = None, # Will be ignored, embedding generated from query_text
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries chunks by semantic similarity using FAISS.
        """
        if not self.memories_data or self.index.ntotal == 0:
            return []

        embedding_to_search = self._generate_embedding(query_text)

        # D: distances, I: indices (FAISS internal IDs in our case due to IndexIDMap)
        # k is the number of nearest neighbors to search for
        k = min(limit, self.index.ntotal) # Cannot search for more items than in index
        distances, faiss_ids = self.index.search(embedding_to_search, k)

        results = []
        # Create a reverse map from faiss_id to memory_id for easier lookup
        faiss_id_to_memory_id = {v: k for k, v in self.memory_id_to_faiss_id.items()}

        for i in range(faiss_ids.shape[1]): # Iterate through returned neighbors
            f_id = faiss_ids[0, i]
            dist = distances[0, i]
            memory_id = faiss_id_to_memory_id.get(f_id)
            
            if memory_id and memory_id in self.memories_data:
                memory_info = self.memories_data[memory_id]
                results.append({
                    "id": memory_id,
                    "text": memory_info["text"],
                    "timestamp": memory_info["timestamp"],
                    "tags": memory_info.get("tags", []),
                    "url": memory_info.get("url"),
                    "similarity_score": float(1 - dist) # Convert L2 distance to a similarity score (0-1 range, higher is better)
                                                      # This is a simple transformation, cosine similarity might be more intuitive if index was normalized
                })
        
        # Sort by similarity score descending as FAISS might not return them perfectly sorted by our score logic
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific memory by its ID."""
        if memory_id in self.memories_data:
            data = self.memories_data[memory_id]
            return {
                "id": memory_id,
                "text": data["text"],
                "timestamp": data["timestamp"],
                "tags": data.get("tags", []),
                "url": data.get("url")
            }
        return None

    def list_all_memories(self) -> List[Dict[str, Any]]:
        """Lists all memories in the database. For debugging/testing."""
        return [
            {
                "id": mid,
                "text": data["text"],
                "timestamp": data["timestamp"],
                "tags": data.get("tags", []),
                "url": data.get("url")
            } for mid, data in self.memories_data.items()
        ]

    def query_memories(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        similarity_query_text: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries memories based on time range and/or similarity.
        - If only similarity_query_text, delegates to query_memories_by_similarity.
        - If no parameters, delegates to query_memories_by_recency.
        - If time filters are present, filters by time first.
        - If time filters AND similarity_query_text are present, performs similarity search on time-filtered results.
        """

        # Case 1: Only similarity query (no time filter)
        if similarity_query_text and not start_time and not end_time:
            return self.query_memories_by_similarity(query_text=similarity_query_text, limit=limit)

        # Case 2: No similarity query and no time filter (default to recency)
        if not similarity_query_text and not start_time and not end_time:
            return self.query_memories_by_recency(limit=limit)

        # Case 3 & 4: Time filter is present (with or without similarity query)
        
        time_filtered_memories_data: List[Dict[str, Any]] = []
        
        for mem_id, data in self.memories_data.items():
            passes_time_filter = True
            memory_timestamp = data["timestamp"]
            # Ensure memory_timestamp is timezone-aware (UTC)
            if memory_timestamp.tzinfo is None: # Should be UTC from storage
                memory_timestamp = memory_timestamp.replace(tzinfo=timezone.utc)

            if start_time:
                st = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
                if memory_timestamp < st:
                    passes_time_filter = False
            if end_time and passes_time_filter: # check passes_time_filter to avoid redundant tz conversion
                et = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)
                if memory_timestamp > et:
                    passes_time_filter = False
            
            if passes_time_filter:
                time_filtered_memories_data.append({
                    "id": mem_id,
                    "text": data["text"],
                    "timestamp": data["timestamp"], 
                    "tags": data.get("tags", []),
                    "url": data.get("url")
                })
        
        # Sort by recency by default for the time-filtered list
        time_filtered_memories_data.sort(key=lambda x: x["timestamp"], reverse=True)

        if not time_filtered_memories_data:
            return []

        # Case 3: Time filter AND similarity query
        if similarity_query_text:
            texts_for_similarity = [mem["text"] for mem in time_filtered_memories_data]
            
            if not texts_for_similarity: return []

            temp_embeddings = self.sentence_model.encode(texts_for_similarity, convert_to_numpy=True).astype('float32')
            if temp_embeddings.shape[0] == 0: return []

            temp_index = faiss.IndexFlatL2(self._embedding_dim)
            temp_index.add(temp_embeddings) # FAISS expects np.array of shape (n_vectors, dim)
            
            query_embedding = self._generate_embedding(similarity_query_text) # Shape (1, dim)
            
            k_search = min(limit, temp_index.ntotal)
            if k_search == 0: return []
                
            distances, indices = temp_index.search(query_embedding, k_search)

            results = []
            for i in range(indices.shape[1]): # Iterate through returned neighbors
                original_mem_idx = indices[0, i] # This is the index in time_filtered_memories_data
                dist = distances[0, i]
                original_memory = time_filtered_memories_data[original_mem_idx]
                
                results.append({
                    "id": original_memory["id"],
                    "text": original_memory["text"],
                    "timestamp": original_memory["timestamp"],
                    "tags": original_memory.get("tags", []),
                    "url": original_memory.get("url"),
                    "similarity_score": float(1 - dist) 
                })
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results # Already limited by k_search, which considers the input limit

        # Case 4: Only time filter (no similarity query)
        else: # similarity_query_text is None, but time filters were applied
            return time_filtered_memories_data[:limit]

# Example usage (for testing within this file if run directly)
if __name__ == '__main__':
    import time # For adding slight delays for recency tests
    # Note: uuid is already imported at the top of the file

    print("Starting MemoryDBHandler tests with FAISS and SentenceTransformer...")
    # It might take a few seconds to load the model for the first time
    db = MemoryDBHandler() # Get the singleton instance
    print("MemoryDBHandler initialized.")

    # Clear any existing data from previous test runs
    all_existing_memories = db.list_all_memories()
    if all_existing_memories:
        print(f"Clearing {len(all_existing_memories)} existing memories from previous test setup...")
        for mem in all_existing_memories: # Renamed 'chunk' to 'mem' for clarity
            db.remove_memory(mem['id'])
    initial_mem_count = len(db.list_all_memories())
    initial_faiss_count = db.index.ntotal
    print(f"Initial state: {initial_mem_count} memories, FAISS index size: {initial_faiss_count}")
    assert initial_mem_count == 0, "Initial memory count should be 0 after clearing."
    assert initial_faiss_count == 0, "Initial FAISS count should be 0 after clearing."


    # Test 1: Add memories (with and without tags, with and without URLs)
    print("\n--- Test 1: Adding Memories ---")
    tags1 = ["source:user_input", "topic:weather", "sentiment:positive"]
    memory1_text = "The weather is sunny and warm today."
    memory1_url = "https://example.com/weather/today"
    memory1_id = db.add_memory(memory1_text, tags=tags1, url=memory1_url)
    print(f"Added memory 1: '{memory1_text}' with ID: {memory1_id}, Tags: {tags1}, URL: {memory1_url}")
    time.sleep(0.01) # Ensure timestamp difference

    tags2 = ["source:webscrape", "domain:example.com", "topic:food", "location:San Francisco"]
    memory2_text = "Exploring the best restaurants in San Francisco for delicious Italian food."
    # Adding memory with tags but no URL
    memory2_id = db.add_memory(memory2_text, tags=tags2)
    print(f"Added memory 2: '{memory2_text}' with ID: {memory2_id}, Tags: {tags2}, URL: None")
    time.sleep(0.01)

    memory3_text = "Learning about artificial intelligence and machine learning models."
    memory3_url = "https://example.ai/ml-models"
    # Adding memory with URL but no tags
    memory3_id = db.add_memory(memory3_text, url=memory3_url) 
    print(f"Added memory 3: '{memory3_text}' with ID: {memory3_id}, Tags: [] (default), URL: {memory3_url}")
    time.sleep(0.01)
    
    tags4 = ["source:user_note", "topic:food", "location:San Francisco"]
    memory4_text = "San Francisco has many good places to eat, not just Italian."
    # Adding memory with tags and no URL initially
    memory4_id = db.add_memory(memory4_text, tags=tags4)
    print(f"Added memory 4: '{memory4_text}' with ID: {memory4_id}, Tags: {tags4}, URL: None")


    all_memories = db.list_all_memories()
    print(f"Total memories in DB: {len(all_memories)}")
    print(f"Total vectors in FAISS index: {db.index.ntotal}")
    assert len(all_memories) == 4, f"Test 1 Failed: Expected 4 memories after adding, got {len(all_memories)}."
    assert db.index.ntotal == 4, f"Test 1 Failed: Expected 4 vectors in FAISS index, got {db.index.ntotal}."
    
    # Verify tags for added memories
    mem1_retrieved = db.get_memory(memory1_id)
    assert mem1_retrieved['tags'] == tags1, f"Test 1 Failed: Tags mismatch for memory1. Expected {tags1}, got {mem1_retrieved['tags']}"
    assert mem1_retrieved['url'] == memory1_url, f"Test 1 Failed: URL mismatch for memory1. Expected {memory1_url}, got {mem1_retrieved['url']}"
    mem2_retrieved = db.get_memory(memory2_id)
    assert mem2_retrieved['url'] is None, f"Test 1 Failed: URL for memory2 should be None, got {mem2_retrieved['url']}"
    mem3_retrieved = db.get_memory(memory3_id)
    assert mem3_retrieved['tags'] == [], f"Test 1 Failed: Tags mismatch for memory3 (no tags). Expected [], got {mem3_retrieved['tags']}"
    assert mem3_retrieved['url'] == memory3_url, f"Test 1 Failed: URL mismatch for memory3. Expected {memory3_url}, got {mem3_retrieved['url']}"
    print("Test 1 Passed (Adding memories and basic tag/URL checks).")

    # Test 2: Query by recency
    print("\n--- Test 2: Query Memories by Recency ---")
    recent_memories = db.query_memories_by_recency(limit=2)
    print(f"Most recent 2 memories:")
    for m in recent_memories: print(f"  ID: {m['id']}, Tags: {m['tags']}, URL: {m['url']}, Text: '{m['text'][:30]}...'")
    assert len(recent_memories) == 2, "Test 2 Failed: Recency query limit not respected."
    assert recent_memories[0]["id"] == memory4_id, f"Test 2 Failed: Recency order incorrect. Expected {memory4_id} first."
    assert recent_memories[0]["tags"] == tags4, f"Test 2 Failed: Tags incorrect for most recent memory. Expected {tags4}."
    assert recent_memories[0]["url"] is None, f"Test 2 Failed: URL for most recent memory (mem4) should be None."
    assert recent_memories[1]["id"] == memory3_id, f"Test 2 Failed: Recency order incorrect. Expected {memory3_id} second."
    assert recent_memories[1]["tags"] == [], f"Test 2 Failed: Tags incorrect for second most recent memory. Expected []."
    assert recent_memories[1]["url"] == memory3_url, f"Test 2 Failed: URL for second most recent memory (mem3) incorrect."
    print("Test 2 Passed.")

    # Test 3: Query by similarity
    print("\n--- Test 3: Query Memories by Similarity ---")
    similar_query_food = "Where to find good pasta in SF?"
    print(f"Querying for: '{similar_query_food}'")
    similar_memories_food = db.query_memories_by_similarity(query_text=similar_query_food, limit=2)
    print(f"Memories similar to '{similar_query_food}':")
    found_memory2 = False
    found_memory4 = False
    for m in similar_memories_food:
        print(f"  ID: {m['id']}, Score: {m['similarity_score']:.4f}, Tags: {m['tags']}, URL: {m['url']}, Text: '{m['text'][:50]}...'")
        if m['id'] == memory2_id:
            found_memory2 = True
            assert m['tags'] == tags2, f"Test 3 Failed: Tags for memory2 incorrect in similarity results. Expected {tags2}"
            assert m['url'] is None, f"Test 3 Failed: URL for memory2 should be None in similarity results."
        if m['id'] == memory4_id:
            found_memory4 = True
            assert m['tags'] == tags4, f"Test 3 Failed: Tags for memory4 incorrect in similarity results. Expected {tags4}"
            
    assert len(similar_memories_food) <= 2, "Test 3 Failed: Similarity query limit not respected for food query."
    assert found_memory2 or found_memory4, f"Test 3 Failed: Expected memory2 or memory4 to be highly similar to food query."
    print("Similarity test for food passed (found relevant memories with correct tags and URL).")

    similar_query_ai = "Tell me about AI."
    tags_ai_query = ["topic:AI"] # Example, not used for filtering yet
    print(f"\nQuerying for: '{similar_query_ai}' (Tags for context: {tags_ai_query})")
    similar_memories_ai = db.query_memories_by_similarity(query_text=similar_query_ai, limit=1)
    print(f"Memories similar to '{similar_query_ai}':")
    found_memory3 = False
    for m in similar_memories_ai:
        print(f"  ID: {m['id']}, Score: {m['similarity_score']:.4f}, Tags: {m['tags']}, URL: {m['url']}, Text: '{m['text'][:50]}...'")
        if m['id'] == memory3_id:
            found_memory3 = True
            assert m['tags'] == [], f"Test 3 Failed: Tags for memory3 incorrect in AI similarity results. Expected []"
            assert m['url'] == memory3_url, f"Test 3 Failed: URL for memory3 incorrect in AI similarity results."

    assert len(similar_memories_ai) <= 1, "Test 3 Failed: Similarity query limit not respected for AI query."
    assert found_memory3, f"Test 3 Failed: Expected memory3 to be highly similar to AI query."
    print("Similarity test for AI passed (found relevant memory with correct tags and URL).")
    print("Test 3 Passed.")

    # Test 4: Get a specific memory
    print("\n--- Test 4: Get Specific Memory ---")
    retrieved_memory = db.get_memory(memory1_id)
    print(f"Retrieved memory with ID {memory1_id}: '{retrieved_memory['text'] if retrieved_memory else 'None'}', Tags: {retrieved_memory['tags'] if retrieved_memory else 'N/A'}, URL: {retrieved_memory['url'] if retrieved_memory else 'N/A'}")
    assert retrieved_memory is not None and retrieved_memory["id"] == memory1_id, "Test 4 Failed: Could not retrieve memory by ID."
    assert retrieved_memory['text'] == memory1_text, "Test 4 Failed: Text mismatch for retrieved memory1."
    assert retrieved_memory['tags'] == tags1, f"Test 4 Failed: Tags mismatch for retrieved memory1. Expected {tags1}, got {retrieved_memory['tags']}."
    assert retrieved_memory['url'] == memory1_url, f"Test 4 Failed: URL mismatch for retrieved memory1. Expected {memory1_url}, got {retrieved_memory['url']}."
    
    retrieved_memory_no_tags_or_url = db.get_memory(memory2_id) # memory2 has tags but no URL
    assert retrieved_memory_no_tags_or_url['url'] is None, f"Test 4 Failed: URL for memory2 should be None. Got {retrieved_memory_no_tags_or_url['url']}."
    print("Test 4 Passed.")

    # Test 5: Remove a memory
    print("\n--- Test 5: Remove Memory ---")
    memory_to_remove_id = memory2_id
    memory_to_remove_info = db.get_memory(memory_to_remove_id) # Get info before removing
    print(f"Removing memory ID: {memory_to_remove_id} ('{memory_to_remove_info['text'][:30]}...'), Tags: {memory_to_remove_info['tags']}, URL: {memory_to_remove_info['url']}")
    removal_success = db.remove_memory(memory_to_remove_id)
    print(f"Removal of memory {memory_to_remove_id} successful: {removal_success}")
    assert removal_success, "Test 5 Failed: remove_memory returned False."
    
    all_memories_after_removal = db.list_all_memories()
    print(f"Total memories after removal: {len(all_memories_after_removal)}")
    print(f"Total vectors in FAISS index after removal: {db.index.ntotal}")
    assert len(all_memories_after_removal) == 3, f"Test 5 Failed: Expected 3 memories after removal, got {len(all_memories_after_removal)}."
    assert db.index.ntotal == 3, f"Test 5 Failed: Expected 3 vectors in FAISS after removal, got {db.index.ntotal}."
    assert db.get_memory(memory_to_remove_id) is None, "Test 5 Failed: Removed memory still accessible via get_memory."
    
    # Verify it's not returned in similarity search
    similar_memories_food_after_remove = db.query_memories_by_similarity(query_text=similar_query_food, limit=3)
    found_removed_memory = False
    for m in similar_memories_food_after_remove:
        if m['id'] == memory_to_remove_id:
            found_removed_memory = True
            break
    assert not found_removed_memory, "Test 5 Failed: Removed memory still appearing in similarity search."
    print("Test 5 Passed.")

    # Test 6: Singleton behavior
    print("\n--- Test 6: Singleton Behavior ---")
    db2 = MemoryDBHandler()
    assert db is db2, "Test 6 Failed: MemoryDBHandler is not a singleton."
    
    tags5 = ["source:internal", "topic:fruits", "session:test_run"]
    memory5_text = "Grapes and apples are common fruits."
    memory5_url = "https://example.com/fruits/grapes-apples"
    memory5_id = db2.add_memory(memory5_text, tags=tags5, url=memory5_url) # Add via second handler instance
    print(f"Added memory 5: '{memory5_text}' with ID: {memory5_id}, Tags: {tags5}, URL: {memory5_url} using second handler instance.")
    
    all_memories_final = db.list_all_memories()
    print(f"Total memories in original DB instance: {len(all_memories_final)}")
    print(f"Total vectors in FAISS (original handler): {db.index.ntotal}")
    assert len(all_memories_final) == 4, f"Test 6 Failed: Singleton state not shared correctly (memory count). Expected 4, got {len(all_memories_final)}"
    assert db.index.ntotal == 4, f"Test 6 Failed: Singleton state not shared correctly (FAISS count). Expected 4, got {db.index.ntotal}"
    
    retrieved_mem5_from_db1 = db.get_memory(memory5_id)
    assert retrieved_mem5_from_db1 is not None, "Test 6 Failed: Memory added via db2 not found in db1."
    assert retrieved_mem5_from_db1['tags'] == tags5, f"Test 6 Failed: Tags for memory added via db2 incorrect. Expected {tags5}, got {retrieved_mem5_from_db1['tags']}."
    assert retrieved_mem5_from_db1['url'] == memory5_url, f"Test 6 Failed: URL for memory added via db2 incorrect. Expected {memory5_url}, got {retrieved_mem5_from_db1['url']}."
    print("Test 6 Passed.")

    # Test 7: Remove non-existent memory
    print("\n--- Test 7: Remove Non-existent Memory ---")
    non_existent_id = str(uuid.uuid4())
    removal_attempt = db.remove_memory(non_existent_id)
    print(f"Attempt to remove non-existent ID {non_existent_id} returned: {removal_attempt}")
    assert not removal_attempt, "Test 7 Failed: Removing non-existent memory should return False."
    assert db.index.ntotal == 4, "Test 7 Failed: FAISS index size changed after attempting to remove non-existent ID."
    print("Test 7 Passed.")

    # Test 8: List all memories and check content
    print("\n--- Test 8: List All Memories Comprehensive Check ---")
    all_memories_check = db.list_all_memories()
    assert len(all_memories_check) == 4, "Test 8 Failed: Incorrect number of memories listed."
    
    expected_ids_texts_tags_urls = {
        memory1_id: (memory1_text, tags1, memory1_url),
        memory3_id: (memory3_text, [], memory3_url), # memory2 was removed
        memory4_id: (memory4_text, tags4, None), # memory4 had no URL
        memory5_id: (memory5_text, tags5, memory5_url)
    }
    
    for mem_dict in all_memories_check:
        mem_id = mem_dict['id']
        assert mem_id in expected_ids_texts_tags_urls, f"Test 8 Failed: Unexpected memory ID {mem_id} in list."
        expected_text, expected_tags, expected_url = expected_ids_texts_tags_urls[mem_id]
        assert mem_dict['text'] == expected_text, f"Test 8 Failed: Text mismatch for {mem_id}."
        assert sorted(mem_dict['tags']) == sorted(expected_tags), f"Test 8 Failed: Tags mismatch for {mem_id}. Expected {expected_tags}, got {mem_dict['tags']}"
        assert mem_dict['url'] == expected_url, f"Test 8 Failed: URL mismatch for {mem_id}. Expected {expected_url}, got {mem_dict['url']}"
    print("Test 8 Passed.")

    # Test 9: Advanced Querying with query_memories (including URL checks)
    print("\n--- Test 9: Advanced Querying with query_memories ---")
    # Setup: Clear and add specific memories with known timestamps and URLs
    for mem in db.list_all_memories(): db.remove_memory(mem['id'])
    
    mem_ancient_text = "Ancient philosophy"
    mem_ancient_tags = ["topic:philosophy", "era:ancient"]
    mem_ancient_url = "https://example.com/philosophy/ancient"
    mem_ancient_id = db.add_memory(mem_ancient_text, tags=mem_ancient_tags, url=mem_ancient_url)
    base_time = datetime.now(timezone.utc)
    db.memories_data[mem_ancient_id]['timestamp'] = base_time - timedelta(days=10)

    time.sleep(0.01)
    mem_renaissance_text = "Renaissance art history"
    mem_renaissance_tags = ["topic:art", "era:renaissance"]
    # No URL for this one
    mem_renaissance_id = db.add_memory(mem_renaissance_text, tags=mem_renaissance_tags)
    db.memories_data[mem_renaissance_id]['timestamp'] = base_time - timedelta(days=5)
    
    time.sleep(0.01)
    mem_modern_text = "Modern computer science"
    mem_modern_tags = ["topic:technology", "era:modern"]
    mem_modern_url = "https://example.com/cs/modern"
    mem_modern_id = db.add_memory(mem_modern_text, tags=mem_modern_tags, url=mem_modern_url)
    db.memories_data[mem_modern_id]['timestamp'] = base_time - timedelta(days=1)

    time.sleep(0.01)
    mem_future_text = "Speculations about future technology"
    mem_future_tags = ["topic:technology", "era:future"]
    mem_future_url = "https://example.com/tech/future"
    mem_future_id = db.add_memory(mem_future_text, tags=mem_future_tags, url=mem_future_url)

    print(f"Setup for Test 9: Added 4 memories with varying timestamps and URLs.")

    # Test 9.1: Time filter only (last 7 days)
    print("\nTest 9.1: Time filter (last 7 days)")
    seven_days_ago = base_time - timedelta(days=7)
    recent_past_memories = db.query_memories(start_time=seven_days_ago, limit=5)
    print(f"Found {len(recent_past_memories)} memories from last 7 days:")
    for m in recent_past_memories: print(f"  ID: {m['id']}, URL: {m['url']}, Text: {m['text']}, Timestamp: {m['timestamp']}")
    assert len(recent_past_memories) == 3 # renaissance, modern, future
    assert mem_renaissance_id in [m['id'] for m in recent_past_memories]
    assert db.get_memory(mem_renaissance_id)['url'] is None # Check URL for one of them
    assert mem_future_id in [m['id'] for m in recent_past_memories]
    assert db.get_memory(mem_future_id)['url'] == mem_future_url # Check URL for one of them
    print("Test 9.1 Passed.")

    # Test 9.2: Time filter and similarity
    print("\nTest 9.2: Time filter (last 7 days) + Similarity ('technology')")
    tech_in_past_7_days = db.query_memories(start_time=seven_days_ago, similarity_query_text="technology", limit=5)
    print(f"Found {len(tech_in_past_7_days)} 'technology' memories from last 7 days:")
    for m in tech_in_past_7_days:
        print(f"  ID: {m['id']}, URL: {m['url']}, Text: {m['text']}, Score: {m.get('similarity_score', -1):.4f}")
        if m['id'] == mem_modern_id:
            assert m['url'] == mem_modern_url
        elif m['id'] == mem_future_id:
            assert m['url'] == mem_future_url
    assert len(tech_in_past_7_days) >= 1 and len(tech_in_past_7_days) <= 2
    print("Test 9.2 Passed.")

    # Test 9.3: Similarity only (delegates to query_memories_by_similarity)
    print("\nTest 9.3: Similarity only ('philosophy')")
    philosophy_memories = db.query_memories(similarity_query_text="philosophy", limit=1)
    print(f"Found {len(philosophy_memories)} 'philosophy' memories:")
    for m in philosophy_memories: print(f"  ID: {m['id']}, URL: {m['url']}, Text: {m['text']}, Score: {m.get('similarity_score', -1):.4f}")
    assert len(philosophy_memories) == 1
    assert philosophy_memories[0]['id'] == mem_ancient_id
    assert philosophy_memories[0]['url'] == mem_ancient_url
    print("Test 9.3 Passed.")

    # Test 9.4: No filters (delegates to query_memories_by_recency)
    print("\nTest 9.4: No filters (should be most recent)")
    default_recency_memories = db.query_memories(limit=2)
    print(f"Found {len(default_recency_memories)} recent memories:")
    for m in default_recency_memories: print(f"  ID: {m['id']}, URL: {m['url']}, Text: {m['text']}, Timestamp: {m['timestamp']}")
    assert len(default_recency_memories) == 2
    assert default_recency_memories[0]['id'] == mem_future_id # Most recent
    assert default_recency_memories[1]['id'] == mem_modern_id # Second most recent
    print("Test 9.4 Passed.")

    # Test 9.5: Time filter only (specific range)
    print("\nTest 9.5: Time filter (between 6 and 2 days ago)")
    start_range = base_time - timedelta(days=6) # Includes renaissance
    end_range = base_time - timedelta(days=2)   # Excludes modern, future
    specific_range_memories = db.query_memories(start_time=start_range, end_time=end_range, limit=5)
    print(f"Found {len(specific_range_memories)} memories from specific range:")
    for m in specific_range_memories: print(f"  ID: {m['id']}, URL: {m['url']}, Text: {m['text']}, Timestamp: {m['timestamp']}")
    assert len(specific_range_memories) == 1
    assert specific_range_memories[0]['id'] == mem_renaissance_id
    print("Test 9.5 Passed.")
    
    # Test 9.6: Time filter resulting in no memories for similarity search
    print("\nTest 9.6: Time filter (very old) + Similarity ('technology')")
    very_old_start = base_time - timedelta(days=100)
    very_old_end = base_time - timedelta(days=50)
    no_tech_in_old_range = db.query_memories(start_time=very_old_start, end_time=very_old_end, similarity_query_text="technology", limit=5)
    assert len(no_tech_in_old_range) == 0, f"Expected 0 memories, got {len(no_tech_in_old_range)}"
    print("Test 9.6 Passed.")
    
    # Test 9.7: Time filter with no results
    print("\nTest 9.7: Time filter (future range)")
    future_start = base_time + timedelta(days=1)
    future_end = base_time + timedelta(days=10)
    no_future_memories = db.query_memories(start_time=future_start, end_time=future_end, limit=5)
    assert len(no_future_memories) == 0, f"Expected 0 memories for future range, got {len(no_future_memories)}"
    print("Test 9.7 Passed.")


    print("\nAll tests completed.")
    print("You can now run 'python memory/memory_db.py' to test the vector DB with tagging.")
    print("Make sure you have run 'pip install -r requirements.txt' to install/update dependencies.") 