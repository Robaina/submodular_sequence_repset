
# Wouldn't resulted repset depend on intinial conditions? e.g, which sequence we start the algorithm with.

In the ```accelerated_greedy_algorithm```:


```python
def accelerated_greedy_selection(db, objective, sim, max_evals=float("inf"), diff_approx_ratio=1.0, repset_size=float("inf"), target_obj_val=float("inf")):
    
    print(f'Repset size: {repset_size}')
    
    assert diff_approx_ratio <= 1.0
    repset = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    num_evals = 0
    while (len(repset) < repset_size) and (len(pq) > 1) and (cur_objective < target_obj_val):
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_diff = -pq[0][0]
        num_evals += 1
        if (num_evals >= max_evals) or (((diff - next_diff) / (abs(diff)+0.01)) >= (diff_approx_ratio - 1.0)):
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
            num_evals = 0
        else:
            heapq.heappush(pq, (-diff, seq_id))
    if len(pq) == 1:
        repset.append(pq[0][1])
    return repset
```

We see that the algorithm starts with the first entry in the heapq data structure, but at initialization, all entries in heapq have $diff = -inf$. Hence, the algorithm starts with an arbitrarily selected sequence. Will the final repset be invariant to this first choice or not? I'm guessing not...