import numpy as np


def print_table(rows_labels, columns_labels, data_):
    row_format = "{:>15}" * (len(rows_labels) + 1)

    print(row_format.format("", *columns_labels))

    for team, row in zip(rows_labels, data_):
        print(row_format.format(team, *row))


row_list = ["row 1", "row 2", "row 3"]
column_list = ["col 1", "col 2", "col 3"]
data = np.array([[1, 2, 1],
                 [0, 1, 0],
                 [2, 4, 2]])

print_table(row_list, column_list, data)

