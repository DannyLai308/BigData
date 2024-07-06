import numpy
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis
import sklearn.metrics

def fisher_lda(x_attributes, x_labels):

    # Calculate means for each attribute in their respective classes 0 and 1
    u1 = numpy.mean(x_attributes[x_labels == 1],0)
    u2 = numpy.mean(x_attributes[x_labels == 0], 0)

    # Remove means from classes
    x1mc = x_attributes[x_labels == 1] - u1
    x2mc = x_attributes[x_labels == 0] - u2

    # Covariance matrices
    S1 = numpy.dot(x1mc.T, x1mc)
    S2 = numpy.dot(x2mc.T, x2mc)
    Sw = S1 + S2
    w = numpy.dot(numpy.linalg.inv(Sw), (u1-u2))

    thresh = 0
    slope = -w[0]/w[1]
    y_int = -thresh / w[1]
    return w, thresh, slope, y_int


def main():
    #------------- Question 1 ------------------------------------------------#
    # Load data points from fld.txt
    X = numpy.loadtxt("fld.txt", delimiter=",")
    x_attributes = X[:, :2] # Retrieve first 2 columns as data attributes x1 x2
    x_labels = X[:, -1].astype(int) # Retrieve last column as class label

    # Use Fisher's LDA implementation
    w, thresh, slope, y_int = fisher_lda(x_attributes, x_labels)
    # Print parameters for Fisher's LDA implementation
    print("\nQuestion 1:")
    print("Fisher's LDA Implementation:")
    print(f"  Slope: {slope:.3f}")
    print(f"  y-intercept: {y_int:.3f}\n")

    # Use sklearn's LDA 
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(x_attributes, x_labels)
    slope_sk = -lda.coef_[0][0]/lda.coef_[0][1]
    int_sk1 = -lda.intercept_[0] 
    # Print parameters for sklearn's LDA implementation
    print("scikit-learn LDA:")
    print(f"  Slope: {slope_sk:.3f}")  
    print(f"  Intercept: {int_sk1:.3f}") 

    # Plot data and discriminant lines
    plt.scatter(x_attributes[x_labels == 1][:, 0], x_attributes[x_labels == 1][:, 1], c='r', marker='.')
    plt.scatter(x_attributes[x_labels == 0][:, 0], x_attributes[x_labels == 0][:, 1], c='b', marker='.')
    # Plot Fisher's LDA line
    beginX = x_attributes[:, 0].min()
    endX = x_attributes[:, 0].max()
    beginY = slope * beginX + y_int
    endY = slope * endX + y_int
    plt.plot([beginX, endX], [beginY, endY], 'g--', label='Fisher LDA Implementation')

    # Plot scikit-learn LDA line (decision boundary)
    x_sklearn = numpy.linspace(beginX, endX, 100)
    y_sklearn = lda.decision_function(numpy.c_[x_sklearn, numpy.zeros_like(x_sklearn)])
    plt.plot(x_sklearn, y_sklearn, 'b--', label='scikit-learn LDA')
    #plt.show()

    # Classification and error rate 
    prediction = (numpy.sign(numpy.dot(w,x_attributes.T) + thresh) + 1)/2
    error = numpy.sum(prediction != x_labels)
    error_rate = (error / len(x_labels)) * 100
    print("number of errors = ", error)
    print(f"percentage of incorrectly classified data points = {error_rate}%")

     # Plot misclassified points
    errorIndex = numpy.argwhere(prediction - x_labels)
    Q = numpy.squeeze(x_attributes[errorIndex])
    plt.scatter(Q[:,0],Q[:,1], c = 'g', marker = 'o')


    #------------- Question 2 ------------------------------------------------#
    # Load data points from spam.txt
    Xq2 = numpy.loadtxt("spam.txt", delimiter=",")
    xq2_attributes = Xq2[:, :-1]  # Retrieve all columns (57 attribute columns) except last column
    xq2_labels = Xq2[:, -1].astype(int)  # Retrieve last column (class labels)

    # Initialize minimum error rate 
    error_rate_min = numpy.inf # Sets to positive infinity
    optimal_thresh = None
    optimal_confusion_matrix = None
    print("\nQuestion 2:")

    # Loop through possible thresholds to determine minimum error
    for thresh_q2 in numpy.arange(-1, 1, 0.1):
        # Fisher's LDA implementation
        w_q2, thresh_q2, slope_q2, y_int_q2 = fisher_lda(xq2_attributes, xq2_labels)
        # Classification based on threshold
        prediction_q2 = (numpy.sign(numpy.dot(w_q2, xq2_attributes.T) + thresh_q2) + 1) / 2

        # Error rate calculation
        error_q2 = numpy.sum(prediction_q2 != xq2_labels)
        error_rate_q2 = (error_q2 / len(xq2_labels)) * 100

        # Update minimum error, threshold, and confusion matrix
        if error_rate_q2 < error_rate_min:
            error_rate_min = error_rate_q2
            optimal_thresh = thresh_q2
            optimal_confusion_matrix = sklearn.metrics.confusion_matrix(xq2_labels, prediction_q2)
        
        # Print discriminant line equation for the first 2 attributes
        if len(xq2_attributes[0]) == 2:
            print(f"Discriminant line equation (based on the first 2 attributes): y = {-slope_q2}x + {y_int_q2}")
            print(f"Slope: {slope_q2:.3f}")
            print(f"Intercept: {y_int_q2:.3f}")
        
    # Minimum error results
    print(f"Minimum Error Rate: {error_rate_min} %")
    print(f"Optimal Threshold: {optimal_thresh}")
    print(f"Confusion Matrix:")
    print(optimal_confusion_matrix)

    # Calculate percentage of misclassified points
    percentage_misclassified = (error_q2 / len(xq2_labels)) * 100
    print(f"Percentage of Misclassified Data Points: {percentage_misclassified}%")
    plt.show()







if __name__ == "__main__":
    main()
