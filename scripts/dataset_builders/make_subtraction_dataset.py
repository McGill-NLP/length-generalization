def add_two_numbers_manually(num1: int, num2: int):
    """
    Add two numbers manually using the column method used in grade school.
    """

    num1_digits = [int(d) for d in str(num1)]
    num2_digits = [int(d) for d in str(num2)]

    # Pad the shorter number with zeros
    if len(num1_digits) > len(num2_digits):
        num2_digits = [0] * (len(num1_digits) - len(num2_digits)) + num2_digits
    elif len(num2_digits) > len(num1_digits):
        num1_digits = [0] * (len(num2_digits) - len(num1_digits)) + num1_digits

    # Add the numbers column by column
    carry = 0
    result = []
    for i in range(len(num1_digits) - 1, -1, -1):
        sum = num1_digits[i] + num2_digits[i] + carry
        carry = sum // 10
        result.append(sum % 10)

    # Add the carry if there is one
    if carry > 0:
        result.append(carry)


    return int("".join(str(d) for d in result[::-1]))

if __name__ == '__main__':
    print(subtract_two_numbers_manually(456, 123))