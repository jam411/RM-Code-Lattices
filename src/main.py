def generate_lattice(mod_input):
    symbols = np.zeros(n)
    for i in range(n):
        tmp = 0
        for j in range(num_level + 1):
            tmp += (1 << j) * mod_input[j][i]
        symbols[i] = tmp
    return symbols