import requests

api_key = "b5a8763a3a0b40b8908214902232304"
api_url = "http://api.weatherapi.com/v1/current.json?key={}&q={}&aqi=no"

def get_weather(city):
    response = requests.get(api_url.format(api_key, city))
    weather_info={}

    if response.status_code == 200:
        data = response.json()
        # Extract current weather data
        temp_c = data['current']['temp_c']
        wind_kph = data['current']['wind_kph']
        # humidity = data['current']['humidity']
        visibility_km = data['current']['vis_km']
        precip_mm = data['current']['precip_mm']

        # Print the extracted data
        weather_info['temp'] = temp_c
        weather_info['wind'] = wind_kph
        weather_info['vis'] = visibility_km
        weather_info['precip'] = precip_mm
        return weather_info
    else:
        print("Error fetching data.")