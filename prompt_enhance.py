from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


# Prompt tempplate to tell the model to only generate a better prompt the user writes
PROMPT_TEMPLATE = """
Rewrite this question in a better way with more details. Do not answer the question or ask politely, give orders, keep as much as possible the original meaning of the question. Do not use markdown format. : {question}
"""
def enhance_prompt(user_prompt):
    # Choosing the model that is going to rewrite the user prompt to a better one
    # so far mistral is the best model for this task gemma gives random stuff
    model = Ollama(
        # model="gemma:2b-v1.1",
        model="mistral",
        base_url='http://localhost:11434',
        temperature=0.02,
        # top_k=top_k,
        # top_p=top_p,
        )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=user_prompt)
    print(prompt)

    # Invoking the model to get the enhanced query
    response_text = model.invoke(prompt)
    return response_text


if __name__ == "__main__" :
    user_prompt = "What is tax rate for the invoice that was written on 12/12/2023?"
    enhanced_prompt = enhance_prompt(user_prompt)
    print(enhanced_prompt)