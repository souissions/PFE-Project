from string import Template

#### System #### Defines the AI's behavior and constraints

system_prompt = Template("\n".join([
    "You are an assistant to generate a response for the user.",
    "You will be provided by a set of docuemnts associated with the user's query.",
    "You have to generate a response based on the documents provided.",
    "Ignore the documents that are not relevant to the user's query.",
    "You can applogize to the user if you are not able to generate a response.",
    "You have to generate response in the same language as the user's query.",
    "Be polite and respectful to the user.",
    "Be precise and concise in your response. Avoid unnecessary information.",
]))

#### Document #### Structures how retrieved documents are presented to the AI
document_prompt = Template(
    "\n".join([
        "## Document No: $doc_num",
        "### Content: $chunk_text",
    ])
)

#### Footer #### Instructs the AI to generate the final answer
footer_prompt = Template("\n".join([
    "Based only on the above documents, please generate an answer for the user.",
    "## Question:",
    "$query",
    "",
    "## Answer:",
]))

#Retrieval Phase: The system fetches documents relevant to the user's query.

#Formatting Phase: Each document is formatted using document_prompt.

#Generation Phase: The AI reads the formatted documents + system_prompt + footer_prompt.

