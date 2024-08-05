from semantic_router.layer import RouteLayer
import numpy as np
from typing import List, Optional

class MultipleRoute(RouteLayer):

    def __init__(self, *args, **kwargs):
        super(MultipleRoute, self).__init__(*args, **kwargs)

    def _pick_routes_above_threshold(self, data, threshold):
        """
        Filters the data by score and extracts unique routes above the given threshold.
        """
        # Sort the data by score in descending order
        sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)

        unique_routes_dict = {}

        # Collect routes and their scores that are above the threshold
        for item in sorted_data:
            if item['score'] > threshold and item['route'] not in unique_routes_dict:
                unique_routes_dict[item['route']] = item['score']

        return unique_routes_dict if unique_routes_dict else None
    
    def _pick_top_n_routes(self, data, n):
        """
        Selects the top N routes based on scores.
        """
        # Sort the data by score in descending order
        sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
        
        unique_routes_dict = {}
        
        # Collect routes and their scores while ensuring uniqueness
        for item in sorted_data:
            if item['route'] not in unique_routes_dict:
                unique_routes_dict[item['route']] = item['score']
        
        # Select the top N unique routes
        top_n_routes_dict = {k: unique_routes_dict[k] for k in list(unique_routes_dict.keys())[:n]}
        
        return top_n_routes_dict if top_n_routes_dict else None
    
    def _pick_top_n_routes_recursive(self, vector: np.ndarray, top_n: int, top_k: int) -> List[str]:
        """
        Recursively retrieves routes until at least top_n unique routes are found.
        """
        results = self._retrieve(xq=vector, top_k=top_k)
        unique_routes = set([item['route'] for item in results])
        
        if len(unique_routes) < top_n:
            # Double top_k and call the function recursively
            return self._pick_top_n_routes_recursive(vector, top_n, top_k * 2)
        return self._pick_top_n_routes(results, top_n)

    def get_multiple_routes(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        multiple_threshold: float = 0.5,
        top_n: int = 1,
        strategy: str = 'threshold'
    ):
        """
        Retrieves multiple routes based on the provided strategy ('threshold' or 'top_n').
        
        Parameters:
        - multiple_threshold: The score threshold used when strategy is 'threshold'.
        - top_n: The number of top routes to select when strategy is 'top_n'.
        - strategy: The strategy to use for selecting routes ('threshold' or 'top_n').
        """

        # if no vector provided, encode text to get vector
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = self._encode(text=text)

        results = self._retrieve(xq=np.array(vector), top_k=self.top_k)
        
        if strategy == 'threshold':
            route_choice = self._pick_routes_above_threshold(results, multiple_threshold)
        elif strategy == 'top_n':
            route_choice = self._pick_top_n_routes_recursive(np.array(vector), top_n, self.top_k)
        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Use 'threshold' or 'top_n'.")

        return route_choice if route_choice else []
