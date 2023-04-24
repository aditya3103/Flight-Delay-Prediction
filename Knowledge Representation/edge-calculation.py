import requests
from bs4 import BeautifulSoup
import datetime
from GUI import get_airports

def get_average_travel_time(origin, destination):
    url = f"https://www.flightstats.com/v2/flight-tracker/route/{origin}/{destination}/"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    flight_rows = soup.select('div.multi-flights-list-row')

    # initialize a list to store the durations of the first five flights
    flight_durations = []

    for row in flight_rows[:2]:
        departure_time = row.select_one('h2.departureTimePadding').text.strip()
        arrival_time = row.select_one('div > h2.flights-list-light-text').text.strip()
        
        # convert departure and arrival times to datetime objects
        departure = datetime.datetime.strptime(departure_time, '%H:%M')
        arrival = datetime.datetime.strptime(arrival_time, '%H:%M')
        
        # calculate the duration and add it to the list
        duration = arrival - departure
        flight_durations.append(duration.seconds // 60)
        
    # calculate the average duration of the first 2 flights
    if len(flight_durations)>0:
        average_duration = sum(flight_durations) / len(flight_durations)
    else:
        average_duration = 120

    return average_duration

def build_search_tree():
    airports = get_airports()
    costs = {}
    for origin in airports:
        print(origin)
        for destination in airports:
            if origin != destination:
                key = (origin, destination)
                reverse_key = (destination, origin)
                if reverse_key in costs:
                    # Use the same cost for the reverse key to avoid duplicating calculations
                    costs[key] = costs[reverse_key]
                else:
                    # Calculate the average flight time and store it as the cost
                    average_time = get_average_travel_time(airports[origin]['code'], airports[destination]['code'])
                    if average_time:
                        costs[key] = average_time
                    else:
                        costs[key] = float('inf')

    return costs