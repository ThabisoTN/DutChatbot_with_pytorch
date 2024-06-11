def test(i,j):

    if(i==0):

        return j

    else:

        return i + test(i-1,j)

print(test(4, 7))
    print(a(i),end=" ")