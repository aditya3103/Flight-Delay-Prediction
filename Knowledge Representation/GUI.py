import tkinter as tk
from tkinter import ttk
import requests
import BFS
import a_star

def get_airports():
    url = "https://api.skypicker.com/locations?type=subentity&term=IN&locale=en-US&location_types=airport&active_only=true"
    response = requests.get(url).json()
    airports = {location["city"]["name"]: {"code": location["code"], "latitude": location["location"]["lat"], "longitude": location["location"]["lon"]} for location in response["locations"]}
    airports['Goa']['code']='GOX'
    return airports

# Define a list of airports for the dropdown menus
cities = list(get_airports())
airports = get_airports()

# Define the function to calculate delay
def calculate_delay(origin, destination):
    # Your code to calculate delay here
    return a_star.a_star_search(origin, destination, cities, airports)
    #return BFS.bfs_search(origin, destination, cities)

# Define the function to handle button click
def button_click():
    origin = origin_var.get()
    destination = dest_var.get()
    delay = calculate_delay(origin, destination)
    delay_label.config(text="Delay: {} mins".format(round(delay,2)))

# Create the main window
root = tk.Tk()
root.title("Flight Delay Calculator")
root.geometry("400x200")
root.resizable(False, False)
root.configure(bg="#f2f2f2")

# Create the dropdown menus for origin and destination airports
origin_var = tk.StringVar(value=cities[0])
dest_var = tk.StringVar(value=cities[1])

origin_label = ttk.Label(root, text="Origin:", background="#f2f2f2", foreground="#000000", font=("Helvetica", 12))
origin_label.place(relx=0.3, rely=0.2, anchor="center")

origin_dropdown = ttk.Combobox(root, textvariable=origin_var, values=cities)
origin_dropdown.place(relx=0.7, rely=0.2, anchor="center")

dest_label = ttk.Label(root, text="Destination:", background="#f2f2f2", foreground="#000000", font=("Helvetica", 12))
dest_label.place(relx=0.3, rely=0.4, anchor="center")

dest_dropdown = ttk.Combobox(root, textvariable=dest_var, values=cities)
dest_dropdown.place(relx=0.7, rely=0.4, anchor="center")

# Create the button and delay label
button = ttk.Button(root, text="Calculate Delay", command=button_click)
button.place(relx=0.5, rely=0.6, anchor="center")

delay_label = ttk.Label(root, text="Delay: ", background="#f2f2f2", foreground="#000000", font=("Helvetica", 16))
delay_label.place(relx=0.5, rely=0.8, anchor="center")

# Start the main loop
root.mainloop()