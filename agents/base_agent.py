from abc import ABC, abstractmethod
import llm_service
import web_search_service

class STEEPV_Agent(ABC):
    """
    Abstract base class for STEEPV agents.
    """
    def __init__(self, topics: list):
        self.topics = topics
        self.domain_map = None

    def generate_domain_map(self):
        """
        Generates a domain map of Core, Adjacent, and Peripheral topics.
        """
        if not self.topics:
            print("No topics to generate domain map from.")
            return

        category = self.__class__.__name__.replace("Agent", "")
        self.domain_map = llm_service.generate_domain_map(self.topics, category)

    def scan_for_signals(self):
        """
        Scans the web for weak signals of change.
        """
        if not self.domain_map:
            print("No domain map to scan from.")
            return []

        signals = []
        # Prioritize Peripheral and Adjacent topics
        topics_to_scan = self.domain_map.get("topics", {}).get("Peripheral", {}).get(self.__class__.__name__.replace("Agent", ""), []) + \
                         self.domain_map.get("topics", {}).get("Adjacent", {}).get(self.__class__.__name__.replace("Agent", ""), [])

        for topic in topics_to_scan:
            search_results = web_search_service.search(topic)
            for result in search_results:
                # In a real implementation, you would use an LLM call to evaluate relevance
                signal = {
                    "title": result.get("title", "N/A"),
                    "description": result.get("snippet", "N/A"),
                    "relevance": f"This is relevant to {topic}",
                    "sourceURL": result.get("link", "N/A")
                }
                signals.append(signal)
        return signals
