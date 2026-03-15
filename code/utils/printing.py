def print_title(msg, n=100):
    """
    Prints a title with a centered message.
    :param msg: The message to print.
    :param n: The buffer size.
    """
    print()
    n = (n // 2) * 2
    print((n // 2) * "===")
    t = (n - len(msg)) // 2
    delta = n - 2 * t - len(msg)
    print(("#" + (t-1) * " ") + f'{msg}' + ((t-1 + delta) * " " + "#"))
    print((n // 2) * "===")
