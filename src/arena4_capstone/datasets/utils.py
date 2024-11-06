def turn(*lines: str, role: str = "user") -> str:
    return "\n".join([f"<start_of_turn>{role}", *lines, "<end_of_turn>"])

def user(*lines: str) -> str:
    return turn(*lines, role="user")

def model(*lines: str) -> str:
    return turn(*lines, role="model")

def combine(*prompts: str) -> str:
    return "\n".join(prompts) + "\n<start_of_turn>model\n"