
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pytz
from suncalc import get_position

# define in numpy terms
def ecliptic_plane_ray(theta):
    """
    The direction the sun is coming from in reference to the ecliptic plane basis.
    :param theta: angle of earth's path around the sun (radians) 0 is the spring equinox.
    :return: unit vector [Ecliptic East, Ecliptic North, Ecliptic Up]
    The earth is in the ecliptic east in northern hemisphere's spring.
    The earth is in the ecliptic north in the northern hemisphere's summer.
    """
    return np.array([[-np.cos(theta)],
                     [-np.sin(theta)],
                     [0]])


tilt = 23.5*np.pi/180 # converted to radians
E = np.matrix([[1, 0,            0],
               [0, np.cos(tilt), np.sin(tilt)],
               [0, -np.sin(tilt), np.cos(tilt)]])


def equitorial_plane_ray(theta):
    """
    The direction the sun is coming from in reference to the equitorial plane basis.
    :param theta: the angle of the earth's path around the sun (radians)
    :return: unit vector [Equitorial East, Equitorial North, Equitorial Up]
    """
    return E @ ecliptic_plane_ray(theta)


def lat_zero_ray(theta, t):
    """
    The direction the sun is coming from in reference to someone standing on the equator
    :param theta: the angle of the earth's path around the sun (radians)
    :param t: sidereal time (radians) pi on spring equinox is high noon.
    :return: unit vector [Up, East, North]
    """
    E = np.matrix([[np.cos(t),  np.sin(t), 0],
                      [-np.sin(t), np.cos(t), 0],
                      [0,            0,           1]])
    return E @ equitorial_plane_ray(theta)


def lat_ray(theta, t, lat):
    """
    The direction the sun is coming from in reference to someone standing at a given latitude
    :param theta: the angle of the earth's path around the sun (radians)
    :param t:  sidereal time (radians)
    :param lat: latitude (deg)
    :return:
    """
    latitude = lat*np.pi/180 # converted to radians
    L = np.matrix([[np.cos(latitude),  0, np.sin(latitude)],
               [0,                 1, 0],
               [-np.sin(latitude), 0, np.cos(latitude)]])
    return L @ lat_zero_ray(theta, t)


def south_facing_photo(unit):
    shift = 0
    if unit[2, 0] > 0:
        if unit[1, 0] >= 0:
            shift = -np.pi
        else:
            shift = np.pi
    x = np.arctan(unit[1,0]/unit[2, 0]) + shift
    if unit[2, 0] == 0:
        if unit[1, 0] >= 0:
            x = np.pi/2
        else:
            x = -np.pi/2
    return np.array([[x],
                     [np.arcsin(unit[0,0])]])

# at longitude -117.1611111111
# at 07:32am 3/20 no daylight savings pacific time, sidereal time is
spring_equinox = dt.datetime(2022, 3, 20, 7, 32)
sdrl = (309.0284342896-180)*np.pi/180

for month in np.linspace(0, np.pi/2, 4):

    i_values = []
    u_values = []
    up_values = []
    east_values = []
    north_values = []
    x_values = []
    y_values = []
    for i in np.linspace(0, 2*np.pi, 100):
        i_values.append(i)

        u = lat_ray(month, i, 33)
        u_values.append(u)
        up_values.append(u[0, 0])
        east_values.append(u[1, 0])
        north_values.append(u[2, 0])
        v = south_facing_photo(u)
        x_values.append(v[0,0])
        y_values.append(v[1,0])
    # plt.plot(x_values, y_values)

# plt.plot(i_values, up_values)
# plt.plot(i_values, east_values)
# plt.plot(i_values, north_values)
# plt.plot(i_values, x_values)
# plt.plot(i_values, y_values)
# plt.legend(['up', 'east', 'north', 'x', 'y'])
# plt.legend(['Jun', 'Jul/May', 'Aug/Apr', 'Sep/Mar', 'Oct/Feb', 'Nov/Jan', 'Dec'])
# plt.show()


# print(L @ Q(0) @ E @ solar_plane_ray(0))
# print(L @ Q(np.pi) @ E @ solar_plane_ray(np.pi/2))
# print(L @ Q(np.pi/2) @ E @solar_plane_ray(np.pi))
# print(L @ Q(3*np.pi/2) @ E @solar_plane_ray(3*np.pi/2))

# print(E @ solar_plane_ray(0))
# print(E @ solar_plane_ray(np.pi/2))
# print(E @solar_plane_ray(np.pi))
# print(E @solar_plane_ray(3*np.pi/2))

# convert local time to sidereal time
# https://astronomy.stackexchange.com/questions/29471/how-to-convert-sidereal-time-to-local-time
# http://www.jgiesen.de/astro/astroJS/siderealClock/

lat = 32.7157
lng = -117.1611
start_hour = 5
end_hour = 20

local = pytz.timezone("America/Los_Angeles")
fig, ax = plt.subplots(nrows=1, ncols=1)
vectors = []
for month in range(6, 13):
    x_values_detail = []
    y_values_detail = []
    x_values = []
    y_values = []
    for hour in range(start_hour, end_hour):
        naive = dt.datetime(2022, month, 20, hour, 0)
        local_dt = local.localize(naive, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)
        pos = get_position(utc_dt, lng=lng, lat=lat)
        x_values.append(pos['azimuth'])
        y_values.append(pos['altitude'])
        vectors.append([np.sin(pos['azimuth'])*np.cos(pos['altitude']), np.cos(pos['azimuth']*np.cos(pos['altitude'])), np.sin(pos['altitude'])])
        for minute in range(0, 60):
            naive = dt.datetime(2022, month, 20, hour, minute)
            local_dt = local.localize(naive, is_dst=None)
            utc_dt = local_dt.astimezone(pytz.utc)
            pos = get_position(utc_dt, lng=lng, lat=lat)
            x_values_detail.append(pos['azimuth'])
            y_values_detail.append(pos['altitude'])
    # plt.plot(x_values_detail, y_values_detail, color='white')
    # plt.scatter(x_values, y_values, color='white')

matrix = np.matrix(vectors)
eigenvalues, eigenmatrix = np.linalg.eigh(matrix.T @ matrix)
if eigenvalues[2] != eigenvalues.max() or eigenvalues[0] != eigenvalues.min():
    raise ValueError('values not ascending')
new_values = matrix @ eigenmatrix

new_angles = []
for idx in range(new_values.shape[0]):
    new_angles.append(list(south_facing_photo(new_values[idx, :].T).T[0]))

new_angles = np.matrix(new_angles)
plt.plot(new_angles[:, 0], new_angles[:, 1])

# date1_naive = dt.datetime(2022, 1, 24, 13, 42)
# local_dt1 = local.localize(date1_naive, is_dst=None)
# utc_dt1 = local_dt1.astimezone(pytz.utc)
# pos1 = get_position(utc_dt1, lng=lng, lat=lat)
# plt.scatter([pos1['azimuth']], [pos1['altitude']])
#
# date2_naive = dt.datetime(2022, 1, 24, 16, 14)
# local_dt2 = local.localize(date2_naive, is_dst=None)
# utc_dt2 = local_dt2.astimezone(pytz.utc)
# pos2 = get_position(utc_dt2, lng=lng, lat=lat)
# plt.scatter([pos2['azimuth']], [pos2['altitude']])

# ax = plt.axes()
# ax.axis('off')
# ax.set_facecolor('xkcd:sky blue')
# plt.legend(range(6, 13))
plt.show()
# plt.savefig(fname='sun_plot.png', transparent=True)

# https://www.etsy.com/listing/1039326256/transparency-screen-printing-accurip?click_key=5710106cc358ccd55a1f8d698cf1471fdca55f2c%3A1039326256&click_sum=66a63d5b&ga_order=most_relevant&ga_search_type=all&ga_view_type=gallery&ga_search_query=printed+transparencies&ref=sr_gallery-1-1

