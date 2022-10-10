# Python3 program to check if square root
# of a number under modulo p exists or not

# Utility function to do modular
# exponentiation. It returns (x^y) % p.
def power(x, y, p):
    res = 1  # Initialize result
    x = x % p

    # Update x if it is more than
    # or equal to p
    while (y > 0):

        # If y is odd, multiply
        # x with result
        if (y & 1):
            res = (res * x) % p

        # y must be even now
        y = y >> 1  # y = y/2
        x = (x * x) % p
    return res


# Returns true if there exists an integer
# x such that (x*x)%p = n%p
def squareRootExists(n, p):
    # Check for Euler's criterion that is
    # [n ^ ((p-1)/2)] % p is 1 or not.
    if (power(n, (int)((p - 1) / 2), p) == 1):
        return True
    return False


# Driver Code
p = 19
n = 87463
if (squareRootExists(n, p) == True):
    print("Yes")
else:
    print("No")
    print("Biba")

# This code is contributed by Rajput-Ji