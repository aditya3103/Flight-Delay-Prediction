import math

def haversine_distance(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # calculate the distance in km
    km = 6371 * c

    # calculate the time taken in minutes (assuming an average speed of 800 km/h)
    time_taken = km / 800 * 60

    return time_taken