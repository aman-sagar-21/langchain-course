import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def main():
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and former public official known for his leadership of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$828 billion."""
    summary_template = """ given the information {information} about a person I want you to create:
    1. A short summary 
    2. Two interesting facts about the person
    """
    summary_prompt_template = ChatPromptTemplate.from_template(
        template=summary_template,
    )
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    
    print("SUMMARY HERE:")
    print(response.content)




if __name__ == "__main__":
    main()
