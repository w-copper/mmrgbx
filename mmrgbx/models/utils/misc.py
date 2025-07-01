def multiply_weight(data: dict, weight: float, key_like="loss"):
    for k, v in data.items():
        if key_like in k:
            data[k] = v * weight
    return data
