import math
import numpy as np
from scipy.optimize import minimize


def great_circle_distance(latitudeA, longitudeA, latitudeB, longitudeB):
    # Degrees to radians
    phi1 = math.radians(latitudeA)
    lambda1 = math.radians(longitudeA)

    phi2 = math.radians(latitudeB)
    lambda2 = math.radians(longitudeB)

    delta_lambda = math.fabs(lambda2 - lambda1)

    central_angle = \
        math.atan2 \
                (
                # Numerator
                math.sqrt
                    (
                    # First
                    math.pow
                        (
                        math.cos(phi2) * math.sin(delta_lambda)
                        , 2.0
                    )
                    +
                    # Second
                    math.pow
                        (
                        math.cos(phi1) * math.sin(phi2) -
                        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
                        , 2.0
                    )
                ),
                # Denominator
                (
                        math.sin(phi1) * math.sin(phi2) +
                        math.cos(phi1) * math.cos(phi2) * math.cos(delta_lambda)
                )
            )

    R = 6371.009  # Km
    return R * central_angle


# Mean Square Error
# locations: [ (lat1, long1), ... ]
# distances: [ distance1, ... ]
def mse(x, locations, distances):
    mse = 0.0
    for location, distance in zip(locations, distances):
        distance_calculated = great_circle_distance(x[0], x[1], location[0], location[1])
        mse += math.pow(distance_calculated * 1000 - distance, 2.0)
    return mse / len(distances)


# initial_location: (lat, long)
# locations: [ (lat1, long1), ... ]
# distances: [ distance1,     ... ]
if __name__ == '__main__':
    locations = [(26.583353, 106.726524), (26.571339, 106.74277), (26.56623, 106.72763)]
    distances = [1330, 2930, 1880]
    m_y = sum([i[0] for i in locations]) / 3
    m_x = sum([i[1] for i in locations]) / 3
    # initial_location = np.array([26.582104, 106.71819])
    initial_location = np.array([m_y, m_x])

    result = minimize(
        mse,                         # The error function
        initial_location,            # The initial guess
        args=(locations, distances), # Additional parameters for mse
        method='L-BFGS-B',           # The optimisation algorithm
        options={
            'ftol':1e-5,         # Tolerance
            'maxiter': 1e+7      # Maximum iterations
        })
    location = result.x
    print(result)
    print(location)

    mse_err = mse(location, locations, distances)
    print(mse_err)


