def deci(a, precision=4):
    times = 10 ** precision
    return int(a * times) / times
