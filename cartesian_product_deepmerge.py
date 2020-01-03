import numpy as np

# variable_configs hold the ConfigSpace values which vary over the runs.
# Each key in variable_configs is actually a "key_chain" which is a '.' separated "chain" of keys which represent keys in a nested dict with the rightmost key being the deepest in the hierarchy.
# Made it '.' separated because the logic seemed simpler than to have the entire nested dict for the code further below. Not sure if that's surely the case.
variable_configs = {
"a.b.c": [1, 2, 3],
"d.e": [1, 3],
"d.f": [5],
"f.g": [i for i in range(4)],
}

# fixed_configs hold the ConfigSpace values which don't vary over the runs.
#TODO Just do a for loop over the merge of each of the variable_configs in Cartesian product with the fixed_configs.


def deepmerge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


num_configs_list = []
for key_chain in sorted(variable_configs.keys()):
    num_configs_list.append(len(variable_configs[key_chain]))
num_configs = np.product(num_configs_list)
print("List of no. of configs for each config in variable_config:", num_configs_list, "Total no. of configs. in the Cartesian product of diff. configs in variable_configs:", num_configs)

cartesian_product = [None] * num_configs
for i in range(num_configs):
    cartesian_product[i] = {}
# print(len(cartesian_product), "len(cartesian_product)")
# counter = 0
for i, key_chain in enumerate(sorted(variable_configs.keys())):
    key_chain_split = key_chain.split(".")
    for j, val in enumerate(variable_configs[key_chain]):
        temp_dict = {key_chain_split[-1]: val}
        for key in reversed(key_chain_split[:-1]):
            temp_dict = {key: temp_dict}
        prod_smaller = int(np.product(num_configs_list[i+1:])) # because product() returns float64 for empty array
        prod_bigger = int(np.product(num_configs_list[:i]))
        # print(prod_smaller, prod_bigger)
        curr_list = [j * prod_smaller + np.arange(prod_smaller) + prod_smaller * len((variable_configs[key_chain])) * k for k in range(prod_bigger)]
        curr_list = list(np.concatenate(curr_list))
        # print("curr_list:", list(curr_list))
        # modulus = len(variable_configs[key_chain])
        # print(cartesian_product)
        # print("temp_dict:", temp_dict)
        for l in range(len(curr_list)):
            cartesian_product[curr_list[l]] = deepmerge(cartesian_product[curr_list[l]], temp_dict)
            # cartesian_product[curr_list[l]].update(temp_dict) ####IMP update is "in-place"! Do not use assignment: cartesian_product[curr_list[l]] =; update() is also not a deepmerge!
        # print(i, np.product(num_configs_list[i+1:]))
        # print(cartesian_product[i + np.product(num_configs_list[i+1:])])
        # cartesian_product[i + np.product(num_configs_list[i+1:])] = 1#cartesian_product[i + np.product(num_configs_list[i+1:])].update(temp_dict)

print(cartesian_product)



# TODO recursively merge dicts: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge/7205107#7205107
# counter // 1 % 4 = j
# for k in range(len(variable_configs[key_chain])):
#     curr_list = [k * prod_smaller + np.arange(prod_smaller) + prod_smaller * len((variable_configs[key_chain])) * l for l in range(prod_bigger)]
# 0, 4, 8,
# 1, 5, 9
# 0, 1, 2, 3, 8, 9, 10, 11,
# 4,5,6,7, 16,17,18,19
# 3
