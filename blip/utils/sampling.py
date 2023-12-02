"""
Implementation of the blip model using pytorch
"""


def farthest_point_sampling(
    position,
    number_of_points
):
    if number_of_points is None:
        return position
