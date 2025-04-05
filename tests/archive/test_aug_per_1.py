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
        """Set up a mock LLM with a simulated response time."""
        self.mock_llm = MagicMock()

        # Simulated valid JSON response
        self.mock_response = json.dumps({
            "calories": 500, "protein": 30, "carbs": 50, "fat": 10,
            "items": ["rice", "chicken", "olive oil"],
            "factoids": ["Rice has been cultivated for over 10,000 years.", 
                         "Olive oil was called liquid gold by the ancient Greeks."]
        })

        self.mock_llm.invoke.side_effect = lambda query: self.mock_response

    ### ✅ Latency Tests ✅ ###
    
    def test_nutrition_chain_latency(self):
        """Ensure the nutrition chain executes within 0.5 seconds."""
        chain = generate_nutrition_info_chain(self.mock_llm)

        start_time = time.time()
        chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")

    def test_shopping_list_chain_latency(self):
        """Ensure the shopping list chain executes within 0.5 seconds."""
        chain = generate_shopping_list_chain(self.mock_llm)

        start_time = time.time()
        chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")

    def test_factoids_chain_latency(self):
        """Ensure the factoids chain executes within 0.5 seconds."""
        chain = generate_factoids_chain(self.mock_llm)

        start_time = time.time()
        chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
        end_time = time.time()

        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5, f"Execution took too long: {execution_time:.3f}s")

    ### ✅ Scalability Test ✅ ###
    
    def test_nutrition_chain_bulk_requests(self):
        """Ensure the nutrition chain can handle 100 requests under 5 seconds."""
        chain = generate_nutrition_info_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5, f"Bulk execution took too long: {execution_time:.3f}s")

    def test_shopping_list_chain_bulk_requests(self):
        """Ensure the shopping list chain can handle 100 requests under 5 seconds."""
        chain = generate_shopping_list_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        end_time = time.time()
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5, f"Bulk execution took too long: {execution_time:.3f}s")

    def test_factoids_chain_bulk_requests(self):
        """Ensure the factoids chain can handle 100 requests under 5 seconds."""
        chain = generate_factoids_chain(self.mock_llm)
        start_time = time.time()

        for _ in range(100):
            chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

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
                chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Memory used: {current / 10**6:.2f} MB, Peak: {peak / 10**6:.2f} MB")
        
        # Ensure memory does not exceed 50MB during execution
        self.assertLess(peak / 10**6, 50, "Memory usage is too high")

if __name__ == "__main__":
    unittest.main()
