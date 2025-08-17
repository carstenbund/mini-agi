from typing import List, Optional
from symbolic_recursion.core.motif import MotifNode, SymbolicMemoryCore
from symbolic_recursion.core.ollama_interface import query_ollama
from symbolic_recursion.utils.id_gen import generate_id
from datetime import datetime

class ChatThread:
    def __init__(self, name: str, model: str = "llama2"):
        self.name = name
        self.model = model
        self.history: List[str] = []

    def ask(self, prompt: str) -> str:
        self.history.append(f"[{datetime.utcnow().isoformat()}] USER: {prompt}")
        resp = query_ollama(prompt, model=self.model)
        self.history.append(f"[{datetime.utcnow().isoformat()}] ASSISTANT: {resp}")
        return resp

class ThreadManager:
    def __init__(self, smc: SymbolicMemoryCore):
        self.smc = smc
        self.threads: List[ChatThread] = []

    def new_thread(self, name: str, model: str = "llama2") -> ChatThread:
        t = ChatThread(name=name, model=model)
        self.threads.append(t)
        return t

    def capture_as_motif(self, thread: ChatThread, symbols: List[str], content: str) -> MotifNode:
        m = MotifNode(
            id=generate_id(),
            symbols=symbols,
            content=content,
            thread_id=thread.name
        )
        self.smc.add_motif(m)
        return m

    def list_threads(self) -> List[str]:
        return [t.name for t in self.threads]
