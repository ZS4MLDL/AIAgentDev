import asyncio
import os
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig
from dotenv import load_dotenv

load_dotenv()

class MySeenMoviesDatabase:
    
    def __init__(self):
        # Initialize with a list of movies the user has watched
        self.seen_movies = [
            "The Matrix", 
            "The Matrix Reloaded", 
            "The Matrix Revolutions", 
            "The Matrix Resurrections"
        ]
    
    @kernel_function(
        name="LoadSeenMovies",
        description="Loads a comma-separated list of movies the user has already seen."
    )
    def load_seen_movies(self) -> str:
        # Join the list into a single string
        return ", ".join(self.seen_movies)


kernel = sk.Kernel()

#Add Service
kernel.add_service(
    sk_oai.OpenAIChatCompletion(
        service_id="chat-gpt",
        ai_model_id="gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
    )
)

# Instantiate the database and import it as a plugin into SK
seen_db = MySeenMoviesDatabase()
seen_movies_plugin = kernel.add_plugin(seen_db, plugin_name="SeenMoviesPlugin")

# We can now access the native function via the kernel if needed
load_seen_movies_func = seen_movies_plugin["LoadSeenMovies"]

print(asyncio.run(load_seen_movies_func(kernel, KernelArguments())))  

# Define a new prompt that incorporates the seen movies via the native function
recommend_prompt_with_memory = """
You are a wise movie recommender tasked with recommending a movie the user has not seen yet.
Below is the list of movies the user has watched:
Movie List: {{SeenMoviesPlugin.LoadSeenMovies}}.

Recommend a new movie that is **not** in the above list, but that the user would likely enjoy based on those movies.
"""
prompt_config_mem = PromptTemplateConfig(
    template=recommend_prompt_with_memory,
    prompt_template_format="semantic-kernel"
)
recommend_function_mem = kernel.add_function(
    prompt_template_config=prompt_config_mem,
    function_name="RecommendNewMovie",
    plugin_name="Recommender"
)


async def test_recommendation_with_memory():
    result = await kernel.invoke(recommend_function_mem, KernelArguments(),service_id="chat-gpt")
    print(result)

asyncio.run(test_recommendation_with_memory())