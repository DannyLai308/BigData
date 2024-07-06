import numpy
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis

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
    # Load data points from fld.txt
    X = numpy.loadtxt("fld.txt", delimiter=",")
    x_attributes = X[:, :2] # Retrieve first 2 columns as data attributes x1 x2
    x_labels = X[:, -1].astype(int) # Retrieve last column as class label

    # Use Fisher's LDA implementation
    w, thresh, slope, y_int = fisher_lda(x_attributes, x_labels)
    # Print parameters for Fisher's LDA implementation
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

    # Classification and error rate 
    prediction = (numpy.sign(numpy.dot(w,x_attributes.T) + thresh) + 1)/2
    error = numpy.sum(prediction != x_labels)
    error_rate = (error / len(x_labels)) * 100
    print("number of errors = ", error)
    print(f"percentage of incorrectly classified data points = {error_rate}%")

     # Plot misclassified points
    # misclassified_pts = X[prediction != x_labels]
    # plt.scatter(misclassified_pts[:, 0], misclassified_pts[:, 1], c='g', marker='o', label='Misclassified Points')
    plt.show()



if __name__ == "__main__":
    main()
