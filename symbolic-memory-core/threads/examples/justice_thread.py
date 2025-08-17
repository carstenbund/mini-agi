# Example symbolic thread
from core.ollama_interface import query_ollama

if __name__ == "__main__":
    prompt = "Explore the concept of Justice as a Gradient."
    response = query_ollama(prompt)
    print("Justice Thread Response:\\n", response)
