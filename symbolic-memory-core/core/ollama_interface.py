import subprocess

def query_ollama(prompt, model="llama2"):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate(input=prompt)
    return stdout
