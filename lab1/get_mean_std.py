# make imports here


# define your functions here

if __name__ == "__main__":
	# your stuff goes here


	# X: all input data
	# y: labels
	# digit_k: integer from 0 to 9 which denotes the digit
	mean_digit_k, std_digit_k = digit_mean_std(X, y, digit_k)
	return mean_digit_k, std_digit_k
