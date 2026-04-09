"""Prompt templates for LM-based classification of MADLAD data."""

# fmt: off
TOPIC_LABELS = {
    "Adult": "Adult/explicit content",
    "Art & Design": "Encompasses architecture and creative fields",
    "Software Dev.": "Software development, programming",
    "Crime & Law": "Law enforcement; excludes financial crime (Finance & Business) and legislative processes (Politics)",
    "Education & Jobs": "Pedagogy, training, certification, academia",
    "Hardware": "Computer/tech hardware",
    "Entertainment": "Movies, music, TV, celebrities",
    "Social Life": "Social interactions, relationships, community",
    "Fashion & Beauty": "Fashion, cosmetics, personal style",
    "Finance & Business": "Financial services, business, corporate; includes financial crime",
    "Food & Dining": "Food, cooking, restaurants, recipes",
    "Games": "Video games, board games, gaming",
    "Health": "Medical, wellness, fitness health topics",
    "History": "Historical events, periods, figures",
    "Home & Hobbies": "Home improvement, DIY, crafts, hobbies",
    "Industrial": "Manufacturing, mining, agriculture, utilities",
    "Literature": "Books, poetry, literary works",
    "Politics": "Government, legislation, political processes",
    "Religion": "Faith, spirituality, religious organizations",
    "Science & Tech.": "Scientific research, technology; subsumes mathematics",
    "Software": "Software products, apps, tools (non-dev)",
    "Sports & Fitness": "Athletic activities, sports events",
    "Transportation": "Vehicles, transit, logistics",
    "Travel": "Tourism, destinations, travel planning",
}

FORMAT_LABELS = {
    "Academic Writing": "Scholarly papers, research articles",
    "Content Listing": "Lists of links, directories, indexes",
    "Creative Writing": "Fiction, poetry, creative prose",
    "Customer Support": "Help pages, support tickets, contact info",
    "Comment Section": "User comments, discussion forums",
    "FAQ": "Frequently asked questions pages",
    "Truncated": "Incomplete/cut-off content",
    "Knowledge Article": "Encyclopedia-style informational articles",
    "Legal Notices": "Terms of service, privacy policies, legal docs",
    "Listicle": "Numbered/bulleted list articles",
    "News Article": "Journalism, news reporting",
    "Nonfiction Writing": "Essays, opinion pieces, nonfiction prose",
    "About (Org.)": "Organization about/info pages",
    "News (Org.)": "Organization press releases, news updates",
    "About (Pers.)": "Personal bio/about pages",
    "Personal Blog": "Personal blog posts",
    "Product Page": "E-commerce product descriptions",
    "Q&A Forum": "Question and answer format (Stack Overflow style)",
    "Spam / Ads": "Spam, advertisements, promotional junk",
    "Structured Data": "Tables, databases, structured content",
    "Documentation": "Technical docs, API references, manuals",
    "Audio Transcript": "Transcribed audio/video content",
    "Tutorial": "How-to guides, tutorials; includes cooking recipes",
    "User Review": "User-generated reviews and ratings",
}

SIB200_LABELS = {
    "geography": "Geography, places, locations, countries, regions",
    "science/technology": "Science, technology, engineering, research",
    "entertainment": "Entertainment, media, arts, culture",
    "politics": "Politics, government, policy, international relations",
    "health": "Health, medicine, wellness, public health",
    "sports": "Sports, athletics, competitions, fitness",
    "travel": "Travel, tourism, destinations, transportation",
}
# fmt: on


def _format_label_list(labels: dict[str, str]) -> str:
    return "\n".join(f'- "{name}": {desc}' for name, desc in labels.items())


CLASSIFY_SYSTEM_PROMPT = "You are a web document classifier. You will be given a document and must classify it along three dimensions: topic, format, and SIB-200 category. Respond with JSON only, no explanation."

CLASSIFY_USER_PROMPT = """\
Classify the following document into exactly one label per dimension.

## Topic categories (pick one)
{topic_labels}

## Format categories (pick one)
{format_labels}

## SIB-200 categories (pick one)
{sib200_labels}

## Document
{text}

Respond with JSON only:
{{"topic": "...", "format": "...", "sib200": "..."}}"""


def build_classify_prompt(text: str) -> str:
    return CLASSIFY_USER_PROMPT.format(
        topic_labels=_format_label_list(TOPIC_LABELS),
        format_labels=_format_label_list(FORMAT_LABELS),
        sib200_labels=_format_label_list(SIB200_LABELS),
        text=text,
    )
