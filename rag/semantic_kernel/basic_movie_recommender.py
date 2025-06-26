import os
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
import asyncio
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template import InputVariable
from semantic_kernel.functions import KernelArguments
from dotenv import load_dotenv

load_dotenv() 

# Define a prompt template with placeholders for context variables
recommend_prompt = """
system:
You have vast knowledge of movies and can recommend anything based on given criteria: the subject, genre, format, and any other custom preference.

user:
Please recommend a {{$format}} about {{$subject}} in the {{$genre}} genre.
Include the following preference: {{$custom}}
"""
# Configure the prompt template and its input variables
prompt_config = PromptTemplateConfig(
    template=recommend_prompt,
    prompt_template_format="semantic-kernel",  # indicates SK formatting
    input_variables=[
        InputVariable(name="format", description="The format to recommend (movie, show, etc.)", is_required=True),
        InputVariable(name="subject", description="The subject or theme of the recommendation", is_required=True),
        InputVariable(name="genre", description="The genre for the recommendation", is_required=True),
        InputVariable(name="custom", description="Any custom preference to refine the recommendation", is_required=True),
    ]
)

# Create a semantic function from the prompt template
kernel = sk.Kernel()

#Add Service
kernel.add_service(
    sk_oai.OpenAIChatCompletion(
        service_id="chat-gpt",
        ai_model_id="gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
    )
)

recommend_function = kernel.add_function(
    prompt_template_config=prompt_config,
    function_name="RecommendMovie",
    plugin_name="Recommender"
)

# Asynchronously invoke the semantic function with specific inputs
async def test_recommendation():
    result = await kernel.invoke(
        recommend_function,
        KernelArguments(
            format="movie", 
            subject="time travel",
            genre="medieval",
            custom="must have comedy elements"
        ),
        service_id="chat-gpt"
    )
    print(result)

# Run the async function 
asyncio.run(test_recommendation())
