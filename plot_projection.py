

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
from suncalc import get_position

local = pytz.timezone("America/Los_Angeles")

lat = 32.7157
lng = -117.1611


def get_vector_here(pst):
    local_dt = local.localize(pst, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    pos = get_position(utc_dt, lng=lng, lat=lat)
    east = np.sin(pos['azimuth']) * np.cos(pos['altitude'])
    north = np.cos(pos['azimuth']) * np.cos(pos['altitude'])
    up = np.sin(pos['altitude'])
    return [east, north, up]

# months: [start hour, end hour]
months = {
    6: [9, 14],
    7: [9, 14],
    8: [9, 14],
    9: [9, 14],
    10: [8, 14],
    11: [7, 13],
    12: [7, 13],
}

vectors = []
for month, hours in months.items():
    x_values_detail = []
    y_values_detail = []
    x_values = []
    y_values = []
    for hour in range(hours[0], hours[1]):
        for minute in range(0, 60):
            vectors.append(get_vector_here(dt.datetime(2022, month, 20, hour, minute)))

matrix = np.array(vectors)
eigenvalues, eigenmatrix = np.linalg.eigh(matrix.T @ matrix)
if eigenvalues[2] != eigenvalues.max() or eigenvalues[0] != eigenvalues.min():
    raise ValueError('values not ascending')
new_values = matrix @ eigenmatrix

azimuth_shifted = np.arctan(new_values[:, 1]/new_values[:, 2])
altitude_shifted = np.arcsin(new_values[:, 0])

fig, ax = plt.subplots()

i = 0
for month, hours in months.items():
    minutes = 60 * (hours[1] - hours[0])
    plt.plot(azimuth_shifted[i: i+minutes], altitude_shifted[i: i+minutes])
    j = 0
    for hour in range(hours[0], hours[1]):
        index = i + j
        plt.text(x=azimuth_shifted[index], y=altitude_shifted[index], s=hour,
                 fontdict=dict(color='black', size = 10))
        j += 60
    i += minutes

# plot other dates
dates = [dt.datetime(2022, 1, 26, 8, 1),
         dt.datetime(2022, 1, 26, 10, 1),
         dt.datetime(2022, 1, 26, 12, 0),
         dt.datetime(2022, 11, 1, 10, 25),]

other_vectors = []
for date in dates:
    other_vectors.append(get_vector_here(date))
other_new_values = np.array(other_vectors) @ eigenmatrix

other_azimuth_shifted = np.arctan(other_new_values[:, 1]/other_new_values[:, 2])
other_altitude_shifted = np.arcsin(other_new_values[:, 0])
plt.scatter(other_azimuth_shifted, other_altitude_shifted)

# plt.legend(months)
fig.set_size_inches(13, 10)
ax.axis('off')
ax.set_aspect('equal')
# plt.show()


plt.savefig(fname='sun_plot_position1.png', transparent=True)
