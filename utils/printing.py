def print_title(msg, n=100):
    print()
    n = (n // 2) * 2
    print((n // 2) * "===")
    t = (n - len(msg)) // 2
    delta = n - 2 * t - len(msg)
    print(("#" + (t-1) * " ") + f'{msg}' + ((t-1 + delta) * " " + "#"))
    print((n // 2) * "===")
