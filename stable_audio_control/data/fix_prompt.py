def make_prompt(tags_str: str) -> str:
    tokens = []
    for t in tags_str.split("\t"):
        t = t.strip()
        if "---" in t:
            tokens.append(t.split("---", 1)[1])
        else:
            tokens.append(t)
    return "piano music, " + ", ".join(tokens)
