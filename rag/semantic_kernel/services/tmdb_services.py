import requests
from semantic_kernel.functions import kernel_function

class TMDbService:
    
    """Semantic service layer for The Movie Database (TMDB) API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key  # Store TMDB API key
    
    @kernel_function(
        name="get_movie_genre_id",
        description="Get the TMDB genre ID for a given movie genre name."
    )
    def get_movie_genre_id(self, genre_name: str) -> str:
        """Lookup genre ID by genre name using TMDB API."""
        print(f"[TMDBService] get_movie_genre_id called with: {genre_name}")
        base_url = "https://api.themoviedb.org/3"
        endpoint = f"{base_url}/genre/movie/list?api_key={self.api_key}&language=en-US"
        response = requests.get(endpoint)
        if response.status_code != 200:
            return ""  # API call failed
        genres = response.json().get('genres', [])
        # Find the genre (case-insensitive match)
        for genre in genres:
            if genre_name.lower() == genre['name'].lower():
                return str(genre['id'])
        return ""  # not found
    
    @kernel_function(
    name="get_top_movies_by_genre",
    description="Get a comma-separated list of currently playing movie titles for a given genre name (e.g., 'Action', 'Comedy')."
    )
    def get_top_movies_by_genre(self, genre: str) -> str:
        """Retrieve currently playing movies and filter by genre."""
        print(f"[TMDBService] get_top_movies_by_genre called with: {genre}")
        genre_id = self.get_movie_genre_id(genre)
        print(f"Genre Id for genre {genre} is {genre_id}")
        if not genre_id:
            return ""  # Unknown genre
        # Call now_playing movies
        base_url = "https://api.themoviedb.org/3"
        now_playing_url = f"{base_url}/movie/now_playing?api_key={self.api_key}&language=en-US"
        response = requests.get(now_playing_url)
        if response.status_code != 200:
            return ""
        movies = response.json().get('results', [])
        # Filter movies that contain the genre_id in their genre_ids list
        filtered_titles = []
        for movie in movies:
            # TMDB genre_ids are integers; convert to string for comparison
            genre_ids = [str(gid) for gid in movie.get('genre_ids', [])]
            if genre_id in genre_ids:
                filtered_titles.append(movie['title'])
            if len(filtered_titles) == 10:
                break  # only take top 10
        return ", ".join(filtered_titles)
