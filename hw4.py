import requests
import random
import numpy as np
import folium

MAX_TIME = 7200
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1

points = [
    (55.7558, 37.6173, 10),
    (55.7522, 37.6156, 8),
    (55.7597, 37.6071, 7),
    (55.7634, 37.6212, 5),
    (55.7415, 37.6164, 6),
]


def calculate_travel_time(origin, destination, profile="foot"):

#profile: 'foot' (пешком), 'car' (на машине), 'bike' (на велосипеде)

    url = f"http://router.project-osrm.org/route/v1/{profile}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"

    params = {
        "overview": "false",
        "steps": "false"
    }

    print(f"Sending request to OSRM...")
    print(f"URL: {url}")
    print(f"Params: {params}")

    response = requests.get(url, params=params)

    print(f"Response Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error Response: {response.text}")
        return float('inf')

    result = response.json()

    try:
        duration = result['routes'][0]['duration']
        print(f"Calculated travel time: {duration} seconds")
        return duration
    except KeyError as e:
        print(f"Error parsing response: {e}")
        print(f"Full response: {result}")
        return float('inf')

def create_distance_matrix(points):
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                print(f"Calculating travel time between point {i} and point {j}...")
                matrix[i][j] = calculate_travel_time(points[i][:2], points[j][:2])
    return matrix

def fitness(route, distance_matrix, max_time):
    total_weight = sum(points[i][2] for i in route)
    total_time = sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    print(f"Fitness calculation for route {route}: Total Weight = {total_weight}, Total Time = {total_time}")

    if total_time > max_time:
        print("Route exceeds MAX_TIME. Fitness = 0.")
        return 0
    print(f"Route is valid. Fitness = {total_weight}.")
    return total_weight


def generate_population(size, num_points):
    population = []
    for _ in range(size):
        individual = list(range(num_points))
        random.shuffle(individual)
        population.append(individual)
    return population

def selection(population, fitnesses):
    tournament_size = 5
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer >= len(child):
                pointer = 0
            child[pointer] = gene
            pointer += 1
    return child

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

def genetic_algorithm(points, max_time, population_size, generations, mutation_rate):
    num_points = len(points)
    distance_matrix = create_distance_matrix(points)

    print("\nDistance Matrix:")
    print(distance_matrix)

    population = generate_population(population_size, num_points)

    for generation in range(generations):
        fitnesses = [fitness(individual, distance_matrix, max_time) for individual in population]
        best_fitness = max(fitnesses)
        best_route = population[fitnesses.index(best_fitness)]

        print(f"\nGeneration {generation}: Best Fitness = {best_fitness}, Route = {best_route}")

        new_population = []
        for _ in range(population_size):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best_fitness = max(fitness(individual, distance_matrix, max_time) for individual in population)
    best_route = population[fitnesses.index(best_fitness)]
    return best_route, best_fitness

def display_map(route):
    map_center = points[route[0]][:2]
    m = folium.Map(location=map_center, zoom_start=14)

    for i in route:
        folium.Marker(location=points[i][:2], popup=f"Point {i}").add_to(m)

    route_coords = [points[i][:2] for i in route]
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(m)

    m.save("map.html")
    print("Карта сохранена в файл map.html")


if __name__ == "__main__":
    best_route, best_fitness = genetic_algorithm(points, MAX_TIME, POPULATION_SIZE, GENERATIONS, MUTATION_RATE)
    print("\nFinal Results:")
    print(f"Best Route: {best_route}")
    print(f"Best Fitness: {best_fitness}")

    display_map(best_route)