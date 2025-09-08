import config
import json
import requests
import uuid
from requests.exceptions import HTTPError

def _call_responses_api(prompt: str) -> str:
    """
    Helper function to call the custom /v1/responses endpoint.
    """
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config.OPENAI_MODEL,
        "prompt": {
            "id": str(uuid.uuid4()),
            "content": prompt
        }
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=data
    )
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
        print(f"Response body: {response.text}")
        raise

    response_data = response.json()
    
    # The response format can vary. Try to find the text in common structures.
    if "text" in response_data:
        return response_data["text"]
    elif "choices" in response_data and response_data["choices"]:
        # Handle cases where response is like standard chat completions
        choice = response_data["choices"][0]
        if "text" in choice:
            return choice["text"]
        elif "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]

    # As a fallback, return the full response text if the structure is unexpected
    return response.text

def generate_deep_research(topic: str) -> str:
    """
    Takes the user's topic, constructs a detailed prompt,
    calls the LLM API, and returns the full text response.
    """
    prompt = f"Generate a deep research report on the topic: {topic}. Cover various aspects including technology, social impact, economic factors, and potential future developments."
    return _call_responses_api(prompt)

def parse_research(dr_text: str) -> dict:
    """
    Parses the deep research text to extract topics, entities, and concepts.
    Returns a JSON object.
    """
    prompt = f"Parse the following research text and extract the key topics, entities, and concepts. Return the result as a valid JSON object with three keys: 'topics', 'entities', and 'concepts'.\n\n{dr_text}"
    json_string = _call_responses_api(prompt)
    return json.loads(json_string)

def generate_domain_map(topics: list, category: str) -> dict:
    """
    Generates a domain map of Core, Adjacent, and Peripheral topics.
    """
    prompt = f"Given the STEEPV category '{category}' and the list of topics {topics}, generate a domain map with 'Core', 'Adjacent', and 'Peripheral' topics. Return the result as a valid JSON object."
    json_string = _call_responses_api(prompt)
    return json.loads(json_string)
