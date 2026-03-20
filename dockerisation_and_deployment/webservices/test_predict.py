from predict import PredictionPipeline

pipeline = PredictionPipeline()

sample = {
    "country_name": "Ghana",
    "country_code": "GHA",
    "year": 2023,
    "malaria_incidence": 180.5,
    "precipitation_mm": 1200.0,
    "pop_density": 130.0,
    "gdp_per_capita": 2200.0,
    "temp_annual_mean_c": 26.5,
    "temp_growing_season_mean_c": 27.1,
    "history": [
        {
            "country_name": "Ghana",
            "country_code": "GHA",
            "year": 2020,
            "malaria_incidence": 170.0,
            "precipitation_mm": 1100.0,
            "pop_density": 125.0,
            "gdp_per_capita": 2100.0,
            "temp_annual_mean_c": 26.1,
            "temp_growing_season_mean_c": 26.8,
        },
        {
            "country_name": "Ghana",
            "country_code": "GHA",
            "year": 2021,
            "malaria_incidence": 175.0,
            "precipitation_mm": 1150.0,
            "pop_density": 127.0,
            "gdp_per_capita": 2150.0,
            "temp_annual_mean_c": 26.3,
            "temp_growing_season_mean_c": 26.9,
        },
        {
            "country_name": "Ghana",
            "country_code": "GHA",
            "year": 2022,
            "malaria_incidence": 178.0,
            "precipitation_mm": 1180.0,
            "pop_density": 129.0,
            "gdp_per_capita": 2180.0,
            "temp_annual_mean_c": 26.4,
            "temp_growing_season_mean_c": 27.0,
        },
    ],
}

result = pipeline.predict_single(sample)
print(result)