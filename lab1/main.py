

def gcd_extended(num1: int, num2: int):
    if num1 == 0:
        return (num2, 0, 1)
    else:
        div, x, y = gcd_extended(num2 % num1, num1)
    return (div, y - (num2 // num1) * x, x)


if __name__ == '__main__':
    import random
    import time
    from statistics import mean
    random.seed(27)

    N = 1000
    a = 1_000
    times = []
    numbers = [random.randint(a, a + 200) for _ in range(N)]
    for num1 in numbers[:N//2]:
        for num2 in numbers[N//2:]:
            t1 = time.time()
            gcd_extended(num1, num2)
            t2 = time.time()
            times.append((t2-t1))

    print(mean(times))
