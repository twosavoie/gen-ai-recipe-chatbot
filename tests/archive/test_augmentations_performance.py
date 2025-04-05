import unittest
import json
import time
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class TestRecipePerformance(unittest.TestCase):

    def setUp(self):
        """Set up a mock LLM that returns proper JSON responses."""
        self.mock_llm = MagicMock()

        # Define valid JSON response strings
        self.nutrition_response = json.dumps({
            "calories": 500, "protein": 30, "carbs": 50, "fat": 10
        })
        self.shopping_list_response = json.dumps({
            "items": ["rice", "chicken", "olive oil"]
        })
        self.factoids_response = json.dumps({
            "factoids": [
                "Rice has been cultivated for over 10,000 years.",
                "Olive oil was called liquid gold by the ancient Greeks."
            ]
        })

        # Ensure the mock returns a valid JSON string
        self.mock_llm.invoke.side_effect = self.mock_response_function

    def mock_response_function(self, query):
        """Mock function to return appropriate JSON responses."""
        if not isinstance(query, dict) or "text" not in query or not isinstance(query["text"], str):
            return json.dumps({"error": "Invalid input"})

        text = query["text"].lower()
        if "shopping" in text:
            return self.shopping_list_response
        elif "factoid" in text:
            return self.factoids_response
        elif "nutrition" in text:
            return self.nutrition_response
        else:
            return json.dumps({"error": "Unknown query"})  # Default error case

    ### ✅ Latency Tests ✅ ###
    
    def test_nutrition_chain_latency(self):
        """Ensure the nutrition chain executes within 0.5 seconds."""
        chain = generate_nutrition_info_chain(self.mock_llm)

        start_time = time.time()
        result = chain.invoke({"text": "nutrition details for rice"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")
        self.assertEqual(result, self.nutrition_response)

    def test_shopping_list_chain_latency(self):
        """Ensure the shopping list chain executes within 0.5 seconds."""
        chain = generate_shopping_list_chain(self.mock_llm)

        start_time = time.time()
        result = chain.invoke({"text": "shopping list for rice"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")
        self.assertEqual(result, self.shopping_list_response)

    def test_factoids_chain_latency(self):
        """Ensure the factoids chain executes within 0.5 seconds."""
        chain = generate_factoids_chain(self.mock_llm)

        start_time = time.time()
        result = chain.invoke({"text": "factoid about rice"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")
        self.assertEqual(result, self.factoids_response)

    ### ✅ Scalability Test ✅ ###
    
    def test_nutrition_chain_bulk_requests(self):
        """Ensure the nutrition chain can handle 100 requests under 5 seconds."""
        chain = generate_nutrition_info_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            result = chain.invoke({"text": "nutrition details for rice"})
            self.assertEqual(result, self.nutrition_response)

        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5, f"Bulk execution took too long: {execution_time:.3f}s")

    def test_shopping_list_chain_bulk_requests(self):
        """Ensure the shopping list chain can handle 100 requests under 5 seconds."""
        chain = generate_shopping_list_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            result = chain.invoke({"text": "shopping list for rice"})
            self.assertEqual(result, self.shopping_list_response)

        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5, f"Bulk execution took too long: {execution_time:.3f}s")

    def test_factoids_chain_bulk_requests(self):
        """Ensure the factoids chain can handle 100 requests under 5 seconds."""
        chain = generate_factoids_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            result = chain.invoke({"text": "factoid about rice"})
            self.assertEqual(result, self.factoids_response)

        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5, f"Bulk execution took too long: {execution_time:.3f}s")

    ### ✅ Resource Usage Test (Memory & CPU) ✅ ###
    
    def test_memory_usage(self):
        """Measure memory usage during 100 calls to each chain."""
        import tracemalloc  # Python’s built-in memory profiler
        
        tracemalloc.start()
        
        chains = [
            generate_nutrition_info_chain(self.mock_llm),
            generate_shopping_list_chain(self.mock_llm),
            generate_factoids_chain(self.mock_llm)
        ]
        
        for chain in chains:
            for _ in range(100):
                chain.invoke({"text": "test input"})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Memory used: {current / 10**6:.2f} MB, Peak: {peak / 10**6:.2f} MB")
        
        # Ensure memory does not exceed 50MB during execution
        self.assertLess(peak / 10**6, 50, "Memory usage is too high")

if __name__ == "__main__":
    unittest.main()
