import re

def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_leading_tickers(text):
    return re.sub(
        r'^\s*(?:\$[A-Z]{1,6}\s*)+(?:[-:]\s*)?',
        '',
        text
    )

def remove_xml_tags(text: str) -> str:
    # Remove all <...> tags, e.g., <Sync id="L313"/>
    import re
    return re.sub(r'<[^>]+>', '', text)

def remove_disclaimer(text: str) -> str:
    disclaimer_keywords = [
        "Disclaimer",
        "Copyright",
        "Definitions",
        "THE INFORMATION CONTAINED IN EVENT TRANSCRIPTS",
        "Users are advised",  # sometimes appears in the last paragraph
    ]
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if any(kw.lower() in line.lower() for kw in disclaimer_keywords):
            break  # stop collecting lines once disclaimer starts
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def clean_transcript_text(text: str) -> str:
    import re

    # 0. Remove XML/HTML-like tags (e.g., <Sync id="L313"/>)
    text = re.sub(r'<[^>]+>', '', text)

    # 1. Remove repeated separators
    text = re.sub(r'[-=]{3,}', '', text)

    # 2. Remove disclaimers at the end
    text = remove_disclaimer(text)

    # 3. Remove top headers / titles and participant lists
    lines = text.split("\n")
    cleaned_lines = []
    skip_keywords = [
        "Corporate Participants",
        "Conference Call Participants",
        "Presentation",
        "Operator",
        "E D I T E D",
        "StreetEvents Event Transcript",
        "Transcripts",
    ]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(kw.lower() in line.lower() for kw in skip_keywords):
            continue
        # Skip participant lists starting with '*'
        if line.startswith("*"):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 4. Remove speaker labels like: Name, Company - Role [number]
    text = re.sub(r'^[^\n]*?, [^\n]*? - [^\n]*? \[\d+\]$', '', text, flags=re.MULTILINE)

    # 5. Remove bracketed references [1], (something)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)

    # 6. Collapse multiple spaces and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()