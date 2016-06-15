import graphlab
import pylab

# load data
sales = graphlab.SFrame('../data/kc_house_data.gl/')

# split into training and testing
train_data,test_data = sales.random_split(.8,seed=0)

# check sizes of lazy sframes
len(test_data), len(train_data)

# closed form solution(compute regression coefficients analytically)
# SArray operations

# Let's compute the mean of the House Prices in King County in 2 different ways.
prices = sales['price'] # extract the price column of the sales SFrame -- this is now an SArray

# recall that the arithmetic average (the mean) is the sum of the prices divided by the total number of houses:
sum_prices = prices.sum()
num_houses = prices.size() # when prices is an SArray .size() returns its length
avg_price_1 = sum_prices/num_houses
avg_price_2 = prices.mean() # if you just want the average, the .mean() function
print "average price via method 1: " + str(avg_price_1)
print "average price via method 2: " + str(avg_price_2)

# generic simple linear regression function
def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    inputSum, outputSum = input_feature.sum(), output.sum()
    # compute the product of the output and the input_feature and its sum
    dot_product = input_feature * output
    dot_product_sum = dot_product.sum()
    # compute the squared value of the input_feature and its sum
    input_squared = input_feature * input_feature
    input_squared_sum = input_squared.sum()
    
    # use the formula for the slope
    n = len(output)
    slope = (dot_product_sum - inputSum * outputSum / n) / (input_squared_sum - inputSum * inputSum / n)
    # use the formula for the intercept
    intercept = outputSum / n - slope * input_feature.mean()
    return (intercept, slope)

test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)


sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = input_feature * slope + intercept
    return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

# plot scatter vs. predicted
pylab.scatter(list(sales['sqft_living']), list(prices))
estimated_prices = get_regression_predictions(sales['sqft_living'], sqft_intercept, sqft_slope)
pylab.plot(list(sales['sqft_living']), list(estimated_prices))
pylab.show()

# residual sum of squares
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predictions = get_regression_predictions(input_feature, intercept, slope) 
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = predictions - output
    # square the residuals and add them up
    RSS = residuals * residuals
    return(RSS.sum())

assert(get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) == 0.0)

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)	

