import numpy
import matplotlib.pyplot as plot
import sklearn.discriminant_analysis

def fisher_lda(x_attributes, x_labels):

    # Calculate means for each attribute in their respective classes 0 and 1
    u1 = numpy.mean(x_attributes[x_labels == 1],0)
    u2 = numpy.mean(x_attributes[x_labels == 0], 0)
    print(u1)
    print(u2)

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
    print(f"  y-intercept: {y_int:.3f}")

    # Use sklearn's LDA 
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(x_attributes, x_labels)
    slope_sk = -lda.coef_[0][0]/lda.coef_[0][1]
    int_sk1 = -lda.intercept_
    # Print parameters for sklearn's LDA implementation
    print("scikit-learn LDA:")
    print(f"  Slope: {slope_sk:.3f}")  
    print(f"  Intercept: {int_sk1:.3f}") 



if __name__ == "__main__":
    main()
