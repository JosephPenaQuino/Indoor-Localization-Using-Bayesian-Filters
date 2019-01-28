"""
Interconnected Embedded System Project
Author: Joseph Peña
"""
# ------------------------------------- Libraries -------------------------------------
import csv
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree as xml


# ------------------------------------- Functions -------------------------------------
def print_list(list_data):
    for value in list_data:
        print(value)


def get_gaussian(bins, mu, sigma):
    result = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2))
    return result


def print_probabilities(data_list, rooms_):
    for ind, val in enumerate(rooms_):
        print(val + ": " + str(data_list[ind]*100) + "%")


def print_table(rows_labels, columns_labels, data):
    # row_format = "{:>25}" * (len(rows_labels) + 1)
    # row_format = "{:>25}" * (10 + 1)
    row_format = "{:>25}" * (len(columns_labels) + 1)

    print("-"*1700)
    print(row_format.format("", *columns_labels))

    for team, row_ in zip(rows_labels, data):
        print(row_format.format(team, *row_))


def print_access_points(rows_labels, columns_labels, data, ap):
    for i, a_p_ in enumerate(ap):
        print("-" * 1700)
        print(" "*40+a_p_)
        print_table(rows_labels, columns_labels, data[i])


# ---------------------------------------- Main ---------------------------------------
# Read CSV files
with open('InputFiles/train_data.csv', 'rt') as f:
    reader = csv.reader(f)
    wifi_data = list(reader)
wifi_data_pre = wifi_data

# Parameters
cell = 0        # name of the place
ssid = 1        # Name of the access point
# bssid = 2       # MAC of the access point
bssid = 1       # MAC of the access point
rssi = 3        # RSS of the access point at the place
date = 4        # Date when the data was taken

# Delete Unwanted Access Point
wifi_data = []
for row in wifi_data_pre:
    # if row[ssid].startswith("UTEC"):
    if True:
        wifi_data.append(row)
wifi_data = np.array(wifi_data)

# Getting Access Point
access_point = wifi_data[:, bssid]
access_point = np.unique(access_point)

# Getting Rooms
rooms = wifi_data[:, cell]
rooms = np.unique(rooms)

# Frequency of scans per room
frequency = np.empty(rooms.size, dtype=int)
for index, room in enumerate(rooms):
    temp = np.array(wifi_data[np.where(wifi_data[:, cell] == room)])
    f = temp[:, date]
    f = np.unique(f)
    frequency[index] = f.size

# Getting Maximum and Minimum RSSI
rssi_vector = wifi_data[:, rssi]
max_rss = np.amax(rssi_vector.astype(np.int))
min_rss = np.amin(rssi_vector.astype(np.int))
# To cover possible coming rssi values
max_rss += 10
min_rss -= 20

# Get Access Point Tables
std = 0
mean = 1
ap_table = np.empty([access_point.size, rooms.size, 2])

for ap_index, ap in enumerate(access_point):
    current_table = np.array(wifi_data[np.where(wifi_data[:, bssid] == ap)])
    for room_index, room in enumerate(rooms):
        current_row = np.array(current_table[np.where(current_table[:, cell] == room)])
        vector = current_row[:, rssi]
        ap_table[ap_index, room_index, std] = np.std(vector.astype(np.int))
        ap_table[ap_index, room_index, mean] = np.mean(vector.astype(np.int))
    # print(str(np.ceil(100*ap_index/access_point.size))+"%")

rss_vector = np.array(list(range(min_rss, max_rss)))

AccessPointsTable = [[[0.0 for x in range(rss_vector.size)] for y in range(rooms.size)] for z in range(access_point.size)]

# Getting Gaussian data
for i, AP in enumerate(access_point):
    for j, cell in enumerate(rooms):
        current_mu = ap_table[i][j][mean]
        if current_mu == 0:
            current_mu = -99
        current_sigma = ap_table[i][j][std]
        if current_sigma == 0:
            current_sigma = 5
        for k, rss in enumerate(rss_vector):
            AccessPointsTable[i][j][k] = get_gaussian(rss, current_mu, current_sigma)
        c_sum = sum(AccessPointsTable[i][j])
        for k, v in enumerate(AccessPointsTable[i][j]):
            AccessPointsTable[i][j][k] = v / c_sum
    # print(str(np.ceil(100 * i / access_point.size)) + "%")

AccessPointsTable = np.array(AccessPointsTable)

# print_list(access_point)
# print_list(rooms)

# print(ap_table[0, 0, mean])

# print_table(rooms, rss_vector, AccessPointsTable[0, :, :])
# print_table(rooms, rss_vector, AccessPointsTable[1, :, :])
# print_table(rooms, rss_vector, AccessPointsTable[2, :, :])

# print_table(rooms, rss_vector, ap_table)
# print_table(rooms, rss_vector, ap_table[1, :, :])
# print_table(rooms, rss_vector, ap_table[2, :, :])

print_access_points(rooms, ['std', 'mean'], ap_table, access_point)

all_data = False
if all_data:
    for i, ap_label in enumerate(access_point):
        for j, room_label in enumerate(rooms):
            plt.plot(rss_vector, AccessPointsTable[i, j, :])
            plt.ylabel('Probability')
            plt.xlabel('RSS')
            plt.title(ap_label+' - '+room_label)
            plt.show()

some_data = False
if some_data:

    # Plotting Aroldo my cuarto rss Gaussian
    plt.plot(rss_vector, AccessPointsTable[0, 0, :])
    plt.ylabel('Probability')
    plt.xlabel('RSS')
    plt.title('AC1 - R1')
    plt.show()
    # Plotting RUTH my cuarto rss Gaussian
    plt.plot(rss_vector, AccessPointsTable[0, 1, :])
    plt.ylabel('Probability')
    plt.xlabel('RSS')
    plt.title('AC1 - R2')
    plt.show()

print_xml = False
if print_xml:
    filename = "OutputFiles/FinalData.xml"
    root = xml.Element("APTables")
    for i, AP in enumerate(access_point):
        AP_table = xml.Element('AP_table', SSID=AP)
        root.append(AP_table)
        # for j in range(max_rss-min_rss+1):
        for j, rss_value in enumerate(rss_vector):
            Rss_column = xml.SubElement(AP_table, "Rss", RSSI=str(rss_value))
            for k, room in enumerate(rooms):
                Room = xml.SubElement(Rss_column, "Room", CELL=room)
                Room.text = str(AccessPointsTable[i][k][j])
                # Room.text = AP+": "+str(j+data.min_rss)+": "+room
    tree = xml.ElementTree(root)

    outFile = open(filename, 'wb')
    tree.write(outFile, pretty_print=True)

Testing_test_data = True
if Testing_test_data:
    Sample = [
        ['test_room_1', 'ac_1', '1', '-99', '2018-08-30 8:08:19'],
        ['test_room_1', 'ac_2', '2', '-81', '2018-08-30 8:08:19'],
        ['test_room_1', 'ac_3', '3', '-68', '2018-08-30 8:08:19']
    ]
    print("------------------ Sample Testing -----------------")
    # ----------------- Getting the room of the max rss ----------
    max_rss_sampled = [-100 for i in range(3)]
    max_rss_wifi = ['' for i in range(3)]

    for index_rss_sampled, value_rss_sampled in enumerate(max_rss_sampled):
        for data_scanned in Sample:
            current_rss = int(data_scanned[rssi])
            if current_rss > max_rss_sampled[index_rss_sampled] and current_rss not in max_rss_sampled:
                max_rss_sampled[index_rss_sampled] = current_rss
                max_rss_wifi[index_rss_sampled] = data_scanned[bssid]

    probabilities = [0.1 / len(rooms) for i in rooms]

    temporal_probabilities = [[0.0 for i in range(len(rooms))] for j in range(3)]

    for i in range(3):
        # Getting the column of rss
        new_array = []
        for AP_index, AP in enumerate(AccessPointsTable):
            current_AP = access_point[AP_index]
            if current_AP == max_rss_wifi[i]:
                for room_index, room in enumerate(AP):
                    new_array.append(room[max_rss_sampled[i] - min_rss])
        # Multiplying
        for index, v in enumerate(new_array):
            temporal_probabilities[i][index] = probabilities[index] * v

        # Normalizing
        normalizer_value = sum(temporal_probabilities[i])
        for index, item in enumerate(temporal_probabilities[i]):
            if normalizer_value == 0:
                temporal_probabilities[i][index] = 0
            else:
                temporal_probabilities[i][index] = temporal_probabilities[i][index] / normalizer_value

    probabilities1 = temporal_probabilities[0]
    probabilities2 = temporal_probabilities[1]
    probabilities3 = temporal_probabilities[2]

    # final_results = [0.0, 0.0, 0.0]
    final_results = [0.0 for i in range(rooms.size)]
    for i in range(len(final_results)):
        final_results[i] = (temporal_probabilities[0][i] +
                            temporal_probabilities[1][i] + temporal_probabilities[2][i]) / 3.0

    print_results = True
    if print_results:
        print("------------------ Sample Testing -----------------")
        print_list(Sample)
        print("------------------ Applying Bayes -----------------")
        print("\n\nInitial array of probabilities:\n")
        print_probabilities(probabilities, rooms)
        print("\n\nArray 1 of probabilities:\n")
        print_probabilities(temporal_probabilities[0], rooms)
        print("\n\nArray 2 of probabilities:\n")
        print_probabilities(temporal_probabilities[1], rooms)
        print("\n\nArray 3 of probabilities:\n")
        print_probabilities(temporal_probabilities[2], rooms)
        print("\n\nFinal array of probabilities:\n")
        print_probabilities(final_results, rooms)


Testing = False
if Testing:
    # --------------------- Testing ----------------------
    Sample = [
        ['My cuarto', ' JARED', 'ec:aa:a0:2e:9f:9b', '-72', '11:45:48'],
        ['My cuarto', 'AROLDO', '84:00:2d:2f:1b:c5', '-77', '11:45:48'],
        ['My cuarto', 'WLAN_BC60', 'd8:fb:5e:ef:bc:65', '-80', '11:45:48'],
        ['My cuarto', 'CUZCANO-5G.', 'a0:38:ee:97:9d:af', '-86', '11:45:48']
    ]

    print("------------------ Sample Testing -----------------")
    print_list(Sample)
    # ----------------- Getting the room of the max rss ----------
    max_rss_sampled = [-100 for i in range(3)]
    max_rss_wifi = ['' for i in range(3)]

    for index_rss_sampled, value_rss_sampled in enumerate(max_rss_sampled):
        for data_scanned in Sample:
            current_rss = int(data_scanned[rssi])
            if current_rss > max_rss_sampled[index_rss_sampled] and current_rss not in max_rss_sampled:
                max_rss_sampled[index_rss_sampled] = current_rss
                max_rss_wifi[index_rss_sampled] = data_scanned[bssid]

    # for i, v in enumerate(max_rss_sampled):
    #     print(max_rss_wifi[i]+": "+str(v))

    # ---------------------- Applying Bayes ------------------
    probabilities = [0.1/len(rooms) for i in rooms]

    temporal_probabilities = [[0.0 for i in range(len(rooms))] for j in range(3)]

    for i in range(3):
        # Getting the column of rss
        new_array = []
        for AP_index, AP in enumerate(AccessPointsTable):
            current_AP = access_point[AP_index]
            if current_AP == max_rss_wifi[i]:
                for room_index, room in enumerate(AP):
                    new_array.append(room[max_rss_sampled[i]-min_rss])
        # Multiplying
        for index, v in enumerate(new_array):
            temporal_probabilities[i][index] = probabilities[index] * v

        # Normalizing
        normalizer_value = sum(temporal_probabilities[i])
        for index, item in enumerate(temporal_probabilities[i]):
            if normalizer_value == 0:
                temporal_probabilities[i][index] = 0
            else:
                temporal_probabilities[i][index] = temporal_probabilities[i][index]/normalizer_value

    probabilities1 = temporal_probabilities[0]
    probabilities2 = temporal_probabilities[1]
    probabilities3 = temporal_probabilities[2]

    # final_results = [0.0, 0.0, 0.0]
    final_results = [0.0 for i in range(rooms.size)]
    for i in range(len(final_results)):
        final_results[i] = (temporal_probabilities[0][i] +
                            temporal_probabilities[1][i] + temporal_probabilities[2][i]) / 3.0

    print_results = True
    if print_results:
        print("------------------ Sample Testing -----------------")
        print_list(Sample)
        print("------------------ Applying Bayes -----------------")
        print("\n\nInitial array of probabilities:\n")
        print_probabilities(probabilities, rooms)
        print("\n\nArray 1 of probabilities:\n")
        print_probabilities(temporal_probabilities[0], rooms)
        print("\n\nArray 2 of probabilities:\n")
        print_probabilities(temporal_probabilities[1], rooms)
        print("\n\nArray 3 of probabilities:\n")
        print_probabilities(temporal_probabilities[2], rooms)
        print("\n\nFinal array of probabilities:\n")
        print_probabilities(final_results, rooms)
