import asyncio
import os
import semantic_kernel as sk
from services.tmdb_services import TMDbService
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai import  OpenAIChatCompletion, OpenAIPromptExecutionSettings
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory

from dotenv import load_dotenv


async def main():

    load_dotenv()

    kernel = sk.Kernel()

    # Initialize the TMDB service with your API key
    tmdb_plugin = kernel.add_plugin(TMDbService(api_key=os.environ["TMDB_API_KEY"]), "TMDBService")

    # Access the functions from the plugin
    genre_id_func = tmdb_plugin["get_movie_genre_id"]
    top_movies_func = tmdb_plugin["get_top_movies_by_genre"]

    #Test the functions
    async def test_tmdb_functions():
        action_id = await genre_id_func(kernel, KernelArguments(genre_name="Action"))
        print(f"Action Genre ID from service: {action_id}")
        top_action = await top_movies_func(kernel, KernelArguments(genre="Action"))
        print(f"Top Action Movies from service: {top_action}")

    
    #await test_tmdb_functions()

    # Add Azure OpenAI chat completion
    chat_completion = OpenAIChatCompletion(
        service_id="chat-gpt",
        ai_model_id="gpt-3.5-turbo",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    kernel.add_service(chat_completion)

    # Add a plugin the LightsPlugin class is defined below)
    
    kernel.add_plugin(
        TMDbService(api_key=os.environ["TMDB_API_KEY"]),
        plugin_name="TMDBService",
    )

    # Enable planning
    execution_settings = OpenAIPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Required(auto_invoke=True)

    # Create a history of the conversation
    history = ChatHistory()
    history.add_system_message("You are a movie recommender assistant.") 

    history.add_user_message("Hi there!")
    history.add_assistant_message("Hello! I can help you find movies to watch. What are you interested in?")
    # Now the actual user query
    history.add_user_message("Can you give me a list of the current top movies for the Action and Comedy genre?")

    # Get the response from the AI
    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel,
    )

    # Print the results
    print("Assistant > " + str(result))

    # Add the message from the agent to the chat history
    history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())