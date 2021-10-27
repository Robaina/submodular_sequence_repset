"""
Utility functions
"""

import random
import heapq


# Either randomly adds or removes elements from a repset
# to match the target size
def random_topup(db, rs, target_repset_size):
    target_repset_size = int(target_repset_size)
    if len(rs) > target_repset_size:
        return random.sample(rs, target_repset_size)
    elif len(rs) < target_repset_size:
        return list(rs) + list(random.sample(set(db.keys()) - set(rs), target_repset_size-len(rs)))
    else:
        return rs

#########################################
# binary_search_repset_size
# ----------------------------
# Performs binary search to find a repset of size
# exactly equal to target.
# Usage:
# db = input database
# f = A function such that f(param) -> repset
# target = target repset size
# param_lower = lowest value of param to check. We assume that higher param <=> larger repset
# param_upper = highest value of param to check
# tolerance = If we find a repset of target<=size<target+tolerance, return sample(repset, target)
# max_iters = maximum binary search iterations to attempt
#########################################
def binary_search_repset_size(db, f, target, param_lower, param_upper, max_iters=10, tolerance=3, final_tolerance=float("inf")):
    # find range that includes target
    # use iters in increments of 2 because we have to compute f() twice at each iter
    for iter_index in range(0,int(float(max_iters)/2),2):
        size_upper = len(f(param_upper))
        size_lower = len(f(param_lower))
        if (size_lower < target and size_upper < target) or (size_lower > target and size_upper > target):
            param_upper += param_upper - param_lower
            param_lower -= param_upper - param_lower
        else:
            break
    range_search_iters = iter_index
    # find target within range
    for iter_index in range(range_search_iters, max_iters):
        try_param = float(param_upper + param_lower) / 2
        rs = f(try_param)
        print (param_lower, param_upper, try_param, len(rs), target)
        if abs(len(rs) - target) <= tolerance:
            ret_rs = rs
            break
        elif len(rs) > target:
            param_upper = try_param
        elif len(rs) < target:
            param_lower = try_param
    if abs(len(rs) - final_tolerance) <= final_tolerance:
        ret_rs = rs
    else:
        len_rs = len(rs)
        raise(Exception("binary_search_repset_size failed to find a suitable param. Final parameters: param_lower={param_lower}; param_upper={param_upper}; len(rs)={len_rs}; target={target}".format(**locals())))
    ret_rs = random_topup(db, ret_rs, target)
    return {"rs": ret_rs, "iters": iter_index+1, "param": try_param}



#########################################
# binary_search_order
# ----------
# Attempts to sample a function y = f(x) as
# uniformly as possible WRT y.
# Usage:
# low_y = f(low_x)
# high_y = f(high_x)
# next = binary_search_order(low_x, low_y, high_x, high_y)
# next_x = (high_x - low_x) / 2
# for i in range(10):
#   next_x, low_x, low_y, high_x, high_y = next(next_x, f(next_x), low_x, low_y, high_x, high_y)
#########################################
def binary_search_order(low_x, low_y, high_x, high_y):
    # pq entry: (-abs(low_y-high_y), low_x, low_y, high_x, low_y)
    pq = []

    def binary_search_order_next(mid_x, mid_y, low_x, low_y, high_x, high_y):

        # Update heap from last time
        heapq.heappush(pq, (-abs(mid_y-low_y), low_x, low_y, mid_x, mid_y))
        heapq.heappush(pq, (-abs(high_y-mid_y), mid_x, mid_y, high_x, high_y))

        # Return next x to try
        negdiff, low_x, low_y, high_x, high_y = heapq.heappop(pq)
        mid_x = float(high_x + low_x) / 2
        return mid_x, low_x, low_y, high_x, high_y

    return binary_search_order_next

def binary_parameter_search(f, low_x, high_x, num_iterations=30):
    low_y = f(low_x)
    if low_y is None:
        raise Exception("low_y is None -- maybe faulty return value from f()")
    high_y = f(high_x)
    get_next = binary_search_order(low_x, low_y, high_x, high_y)
    next_x = (high_x + low_x) / 2
    for i in range(num_iterations):
        next_x, low_x, low_y, high_x, high_y = get_next(next_x, f(next_x), low_x, low_y, high_x, high_y)

