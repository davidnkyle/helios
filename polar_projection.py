

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
    12: [13, 18],
    1: [13, 18],
    2: [13, 18],
    3: [13, 18],
    4: [13, 17],
    5: [14, 17],
    6: [14, 17],
}

vectors = []
for month, hours in months.items():
    year = 2022
    if month == 12:
        year = 2021
    for hour in range(hours[0], hours[1]):
        for minute in range(0, 60):
            vectors.append(get_vector_here(dt.datetime(year, month, 20, hour, minute)))

matrix = np.array(vectors)

center_dt = dt.datetime(2022, 3, 1, 14, 55)
center = np.array(get_vector_here(center_dt))

a1 = np.arctan2(center[1], center[2])
first_twist = np.array([
    [1, 0, 0],
    [0, np.cos(a1), -np.sin(a1)],
    [0, np.sin(a1), np.cos(a1)]
])
a2 = np.arcsin(center[0])
second_twist = np.array([
    [np.cos(a2), 0, -np.sin(a2)],
    [0, 1, 0],
    [np.sin(a2), 0, np.cos(a2)]
])
third_shift = -np.pi/2
polar_center = second_twist @ first_twist @ center
if abs((polar_center**2).sum()-1) > 0.00000001:
    raise ValueError('Not unit vector!')
if abs(polar_center[2] - 1) > 0.00000001:
    raise ValueError('Wrong Polar Center!')


new_values = np.array(matrix @ first_twist.T @ second_twist.T)

theta_shifted = np.arctan2(new_values[:, 0], new_values[:, 1])+third_shift
radius_shifted = np.arccos(new_values[:, 2])

new_x = radius_shifted*np.cos(theta_shifted)
new_y = radius_shifted*np.sin(theta_shifted)

fig, ax = plt.subplots()

i = 0
for month, hours in months.items():
    minutes = 60 * (hours[1] - hours[0])
    j = 0
    for hour in range(hours[0], hours[1]):
        index = i + j
        plt.plot(new_x[index+5: index+56], new_y[index+5: index+56], color='black', alpha=0.5, linewidth=3)
        hour = ((hour-1) % 12)+1
        plt.text(x=new_x[index]-0.01, y=new_y[index]-0.005, s=hour,
                 fontdict=dict(color='black', size = 20))
        j += 60
    i += minutes

# plot other dates
dates = [dt.datetime(2022, 1, 27, 14, 3),
         dt.datetime(2022, 1, 27, 16, 1)
         ]

other_vectors = []
for date in dates:
    other_vectors.append(get_vector_here(date))
other_new_values = np.array(np.array(other_vectors) @ first_twist.T @ second_twist.T)

other_theta_shifted = np.arctan2(other_new_values[:, 0], other_new_values[:, 1]) + third_shift
other_radius_shifted = np.arccos(other_new_values[:, 2])

other_x = other_radius_shifted*np.cos(other_theta_shifted)
other_y = other_radius_shifted*np.sin(other_theta_shifted)

plt.scatter(other_x, other_y, marker="x", color='black', s=300)
# plt.scatter([0], [0])

# plt.legend(months)
fig.set_size_inches(26, 20)
# ax.axis('off')
ax.set_aspect('equal')
plt.show()


# plt.savefig(fname='sun_plot_position4.png', transparent=True)

