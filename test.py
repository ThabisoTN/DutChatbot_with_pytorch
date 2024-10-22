def test(i, j):
    if i == 0:
        return j
    else:
        return i + test(i - 1, j)

# Call the function and print the result
print(test(4, 7))
