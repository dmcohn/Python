# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
David Cohn 
Fall 2015

"""
import sys
import numpy
"""
This function takes in a list of lines from an expression data input file,
partitions the data based on their proper row and column, and inserts the data
into an appropriately sized matrix.

Input:
    expression_file_lines:list of raw lines, with their tags still included,
    from one of the two expression data input files.
Returns: a matrix that contains the expression data, with the tags having
been removed, and the string-based values converted to floats.
"""
def derive_expression_matrix(expression_file_lines):
    expression_file_lines=[x for x in expression_file_lines if x != '\n']
    line=expression_file_lines[0].strip('\n')
    gene_expression_values=line.split('\t')
    gene_expression_matrix=numpy.zeros((len(expression_file_lines),len(gene_expression_values)))
    for i in xrange(0, len(gene_expression_values)):
        gene_expression_matrix[0,i]=float(gene_expression_values[i])
    if(len(expression_file_lines)>1):
        for i in xrange(1,len(expression_file_lines)):
            line=expression_file_lines[i].strip('\n')
            gene_expression_values=line.split('\t')
            for j in xrange(0, len(gene_expression_values)):
                gene_expression_matrix[i,j]=float(gene_expression_values[j])              
    return gene_expression_matrix
"""
This function shuffles the columns of data for the given expression matrix;
This shuffling is necessary in order to randomly divide both the positive
and negative expression matrices into n groups as part of the n-fold cross
validation. Since the shuffle function can only shuffle rows, the function 
transposes rows and columns, shuffles the original columns, and then transposes
the matrix back to its original orientation.

Input:
    matrix: matrix of expression data (either positive or negative phenotype)
Returns: the same matrix, with the columns of data shuffled randomly in order
to be able to perform a random partition of the data into n-groups for cross
validation.
"""
def rearrange_columns(matrix):
    matrix_transposed=numpy.transpose(matrix)
    numpy.random.shuffle(matrix_transposed)
    matrix=(numpy.transpose(matrix_transposed))
    return matrix
"""
This function takes the randomly shuffled expression matrix (either positive
or negative phenotype), and divides that expression matrix into n groups. From
there, the n different groups of positive phenotype and negative phenotype data
will be taken, and one of the n groups of each phenotype will be combined before
starting cross-validation.

Inputs: 
    expression_matrix: expression matrix (either positive or negative phenotype)
with its randomly shuffled columns, corresponding to a different patient.
    n: number of groups that the positive and negative expression matrices will
    be divided into in order to perform n-fold cross validation
Returns: A dictionary of n different lists,  that contain indices of columns
that will be contained within that particular group
"""
def derive_n_groups(expression_matrix, n):
    KNN_group_dictionary={}
    number_of_genes_per_group=expression_matrix.shape[1]/n
    remainder_positive_genes=expression_matrix.shape[1]%n
    for i in xrange (0,n):
        indices_list=[]
        for j in xrange(i*number_of_genes_per_group, (i+1)*number_of_genes_per_group):
            indices_list.extend([j])
        KNN_group_dictionary[str(i)]=indices_list
    for i in xrange(0,remainder_positive_genes):
        indices_list=KNN_group_dictionary[str(i)]
        indices_list.extend([i+n*number_of_genes_per_group])
        KNN_group_dictionary[str(i)]=indices_list
    return KNN_group_dictionary
"""
This function takes in test and training data, that were derived via random partitioning
of the positive and negative phenotype matrices, and for each patient in the test data set,
determines its k nearest neighbors from the training set. From there, the method looks
at the actual classified phenotype of each of these k nearest neighbors, while talling
the total number of appearences of the positive and negative phenotypes for these k vectors.
Finally, after totaling the number of vectors classified as either positive or negative,
the function compares the percentage of positive vectors out of the total number of vectors,
and if that percentage is greater than a minimum user-inputted threshold p, the test data vector
will be classified as having the positive phenotype. Otherwise, if the percentage does not meet
the given threshold, the test data vector will be classified as having the negative phenotype.

Inputs:
    KNN_test_data: testing data containing both positive and negative phentoype vectors;
    the test data encompasses a single group out of the n total groups of data that were
    derived from the user. We will be classifying the test data, not knowing their true
    phenotype, based on the training data.
    KNN_training_data: training data containing positive and negative phenotype vectors;
    in remaining consistent with Leave-One-Out Cross Validation (LOOC Validation), the
    remaining n-1 groups that are not found in the test set will be contained in the training set.
    We will use the known classification of the vectors in the training set, coupled with assessing
    the similarity between a test vector and vectors in the training set using Euclidean distance,
    to make judgements regarding the test vector's phenotype.
    KNN_testing_positive_set: list of indices representing the positive-based segment of the testing set
    KNN_parameters_tuple: tuple that stores the user inputted k, n, and minimum positive threshold
    p values
    positive_training_sample_size: index that represents the partition between the positive and
    negative phenotype vectors in the training data
Returns: a classification matrix, with each of the vectors composing the test set having its own row;
in the first column, the vector's true phenotype is listed (with a positive phenotype represented by a 1,
and a negative phenotype represented by a 0). In the second column, the classification of that test vector,
based on its K-nearest neighbors, is given, using the same 0-1 based system described in the previous sentence.
"""    
def k_nearest_neighbors_determination(KNN_test_data,KNN_training_data,KNN_testing_positive_set, KNN_parameters_tuple, positive_training_sample_size):
    KNN_classification_matrix=numpy.zeros((KNN_test_data.shape[1],2))
    for j in xrange(0,KNN_test_data.shape[1]):
        if(j<len(KNN_testing_positive_set)):
            KNN_classification_matrix[j,0]=1
        else:
            KNN_classification_matrix[j,0]=0
        test_vector=KNN_test_data[:, j]
        Euclidean_distances_matrix=numpy.zeros((KNN_training_data.shape[1],2))
        for l in xrange(0,KNN_training_data.shape[1]):
            Euclidean_distances_matrix[l,0]=numpy.linalg.norm(test_vector-KNN_training_data[:,l])
            if(l<positive_training_sample_size):
                Euclidean_distances_matrix[l,1]=1
            else:
                Euclidean_distances_matrix[l,1]=0
        Euclidean_distances_matrix=Euclidean_distances_matrix[numpy.argsort(Euclidean_distances_matrix[:,0])]
        K_neighbors=Euclidean_distances_matrix[0:KNN_parameters_tuple[0],:]
        number_of_positive_neighbors=0
        for i in xrange(0, K_neighbors.shape[0]):
            number_of_positive_neighbors+=K_neighbors[i,1]
        if(number_of_positive_neighbors/KNN_parameters_tuple[0]>=KNN_parameters_tuple[1]):
            KNN_classification_matrix[j,1]=1
        else:
            KNN_classification_matrix[j,1]=0
    return KNN_classification_matrix
"""
This function conducts the first part of the evaluation process for the k-nearest
neighbors algorithm, by talling the number of True Positives, true negatives, false
positives and false negatives associated with a particular partition of the data 
into training and test sets. For this algorithm, a true positive refers to classifying
a vector as having the positive phenotype, with the vector's actual classification also
being positive. In turn, a true negative refers to classifying a vector as having the negative
phenotype, with vector's actual classification also being negative. Thirdly, a false positive
refers to classifying a vector as having the positive phenotype, with the vector's actual
classification being negative. Finally, a false negative refers to classifying a vector
as having the negative phenotype, when the vector's actual classification is the positive
phenotype.

Inputs:
    KNN_classification_matrix: classification matrix, which contains the actual and KNN-based
    classification of each vector in the testing set, as represented with 0's and 1's.
Returns: A tuple containing the total number of true positives, true negatives, false positives
and false negatives, using the scoring system described above.

"""
def k_nearest_neighbors_algorithm_evaluation(KNN_classification_matrix):
    True_Positives=0
    True_Negatives=0
    False_Positives=0
    False_Negatives=0
    for j in xrange(0, KNN_classification_matrix.shape[0]):
        if(KNN_classification_matrix[j,0]==KNN_classification_matrix[j,1]==1):
            True_Positives+=1
        elif(KNN_classification_matrix[j,0]==KNN_classification_matrix[j,1]==0):
            True_Negatives+=1
        elif(KNN_classification_matrix[j,0]==1 and KNN_classification_matrix[j,1]==0):
            False_Negatives+=1
        elif(KNN_classification_matrix[j,0]==0 and KNN_classification_matrix[j,1]==1):
            False_Positives+=1
    algorithm_quality_assessment_tuple=(True_Positives,True_Negatives, False_Positives, False_Negatives)
    return algorithm_quality_assessment_tuple
"""
This function takes in the total number of true positives, true negatives, false positives,
and false negatives associated with the n runs of n-fold cross validation, and determines the
overall accuracy, sensitivity, and specificity of the KNN algorithm using the following formulas:

Accuracy: (True_Positives+True_Negatives)/(True_Positives+True_Negatives+False_Positives+False_Negatives)
Sensitivity: True_Positives/(True_Positives+False_Negatives)
Specificity: True_Negatives/(True_Negatives+False_Positives)

Input:
    n_fold_cross_validation_results: a matrix, containing the number of True Positives, True Negatives,
    False Positives, and False Negatives for a particular fold of the cross validation procedure. Each
    fold has its own row in the matrix, with the True Positives, True Negatives, False Positives, and 
    False Negatives constituting the columns.
Returns: A tuple containing the accuracy, sensitivity and specificity metrics, across all n-folds,
for the KNN algorithm.
"""
def determine_classification_metrics(n_fold_cross_validation_results):
    Total_True_Positives=sum(n_fold_cross_validation_results[:,0])
    Total_True_Negatives=sum(n_fold_cross_validation_results[:,1])
    Total_False_Positives=sum(n_fold_cross_validation_results[:,2])
    Total_False_Negatives=sum(n_fold_cross_validation_results[:,3])
    Sensitivity=float(Total_True_Positives/(Total_True_Positives+Total_False_Negatives))
    Specificity=float(Total_True_Negatives/(Total_True_Negatives+Total_False_Positives))
    Accuracy=float((Total_True_Positives+Total_True_Negatives)/(Total_True_Positives+Total_True_Negatives+Total_False_Positives+Total_False_Negatives))
    classifier_metrics_tuple=(round(Accuracy,2), round(Sensitivity,2), round(Specificity,2))
    return classifier_metrics_tuple
"""
This method represents the overarching method for performing n-fold cross validation.
The steps associated with this method are as follows:
    1. Create training (n-1 segments) and testing sets (1 segment) composed of both
    positive and negative expression data
    2. Determine k nearest neighbors
    3. Total True Positives, True Negatives, False Positives and False Negatives
    4. Calculate Accuracy, Sensitivity, and Specificity across all n folds
Step #1 is performed in this function, while Steps #2-#4 are performed in submethods above.

Input: 
    positive_file_expression_matrix:expression data matrix for positive phenotype group
    negative_file_expression_matrix:expression data matrix for negative phenotype group
    KNN_group_dictionary_tuple: Dictionary representing the division of the positive and negative
    expression matrices into n groups, with each group having a list of indices that reflect the 
    vectors associated with that group. There is a dictionary for both the positive and negative data
    sets.
    KNN_parameters_tuple: tuple that stores the user inputted k, n, and minimum positive threshold
    p values
Returns: a tuple, derived in submethod determine_classification_metrics, that contains the
accuracy, sensitivity and specificity scores for the knn algorithm across all n folds. 
"""
def performing_n_fold_cross_validation(positive_file_expression_matrix, negative_file_expression_matrix, KNN_group_dictionary_tuple, KNN_parameters_tuple):
    n_fold_cross_validation_results=numpy.zeros((KNN_parameters_tuple[2],4))
    for i in xrange(0,n):
        KNN_positive_group_dictionary=KNN_group_dictionary_tuple[0]
        KNN_negative_group_dictionary=KNN_group_dictionary_tuple[1]
        KNN_testing_positive_set=KNN_positive_group_dictionary[str(i)]
        KNN_testing_negative_set=KNN_negative_group_dictionary[str(i)]
        KNN_test_data=numpy.zeros((positive_file_expression_matrix.shape[0], len(KNN_testing_positive_set)+len(KNN_testing_negative_set)))
        for j in xrange(0,len(KNN_testing_positive_set)):
            KNN_test_data[:,j]=positive_file_expression_matrix[:,KNN_testing_positive_set[j]]
        for k in xrange(0, len(KNN_testing_negative_set)):
            KNN_test_data[:,len(KNN_testing_positive_set)+k]=negative_file_expression_matrix[:,KNN_testing_negative_set[k]]
        KNN_training_data=numpy.zeros((positive_file_expression_matrix.shape[0],positive_file_expression_matrix.shape[1]-len(KNN_testing_positive_set)+negative_file_expression_matrix.shape[1]-len(KNN_testing_negative_set)))
        position=0
        for l in xrange(0, positive_file_expression_matrix.shape[1]):
            if(l not in KNN_testing_positive_set):
                KNN_training_data[:,position]=positive_file_expression_matrix[:,l]
                position+=1    
        positive_training_sample_size=position
        for l in xrange(0, negative_file_expression_matrix.shape[1]):
            if(l not in KNN_testing_negative_set):
                KNN_training_data[:,position]=negative_file_expression_matrix[:,l]
                position+=1
        KNN_classification_matrix=k_nearest_neighbors_determination(KNN_test_data,KNN_training_data,KNN_testing_positive_set, KNN_parameters_tuple, positive_training_sample_size)
        algorithm_quality_assessment_tuple=k_nearest_neighbors_algorithm_evaluation(KNN_classification_matrix)
        for j in xrange(0,4):
            n_fold_cross_validation_results[i,j]=algorithm_quality_assessment_tuple[j]
    classifier_metrics_tuple=determine_classification_metrics(n_fold_cross_validation_results)
    return classifier_metrics_tuple    
"""
This method creates the output file where the results of the KNN cross-validation
analysis will be written to. This output file, titled knn.out, will contain the 
k, n, and minimum positive threshold p values, as well as the accuracy, sensitivity,
and specificity values for the KNN algorithm. 

Inputs:
    KNN_parameters_tuple: tuple that stores the user inputted k, n, and minimum positive threshold
    p values
    classifier_metrics_tuple: a tuple, derived in submethod determine_classification_metrics, that contains the
accuracy, sensitivity and specificity scores for the knn algorithm across all n folds. 

Returns: output_file containing the results of performing n-fold cross validation on
our KNN algorithm, as well as the user-inputted k, n, and minimum positive threshold
p values
"""
def create_output_file(KNN_parameters_tuple, classifier_metrics_tuple):
    output_file=open('knn.out', 'w')
    output_file.write('k: '+KNN_parameters_tuple[0]+'\n')
    output_file.write('p: '+KNN_parameters_tuple[1]+'\n')
    output_file.write('n: '+KNN_parameters_tuple[2]+'\n')
    output_file.write('accuracy: '+classifier_metrics_tuple[0]+'\n')
    output_file.write('sensitivity: '+classifier_metrics_tuple[1]+'\n')
    output_file.write('specificity: '+classifier_metrics_tuple[2]+'\n')
    output_file.close()
    return output_file

positive_file=open(sys.argv[1])
negative_file=open(sys.argv[2])
positive_file_lines=positive_file.readlines()
negative_file_lines=negative_file.readlines()
#Reading input file lines into expression matrix
positive_file_expression_matrix=derive_expression_matrix(positive_file_lines)
negative_file_expression_matrix=derive_expression_matrix(negative_file_lines)
  
k=int(sys.argv[3])
p=round(float(sys.argv[4]),2)
n=int(sys.argv[5])

#shuffle columns
positive_file_expression_matrix=rearrange_columns(positive_file_expression_matrix)
negative_file_expression_matrix=rearrange_columns(negative_file_expression_matrix)

KNN_parameters_tuple=(k,p,n)

#Creating n different positive and negative Groups
KNN_positive_group_dictionary=derive_n_groups(positive_file_expression_matrix, n)
KNN_negative_group_dictionary=derive_n_groups(negative_file_expression_matrix, n)
KNN_group_dictionary_tuple=(KNN_positive_group_dictionary, KNN_negative_group_dictionary)
#n-fold cross validation
classifier_metrics_tuple=performing_n_fold_cross_validation(positive_file_expression_matrix, negative_file_expression_matrix, KNN_group_dictionary_tuple, KNN_parameters_tuple)
K_value=str(KNN_parameters_tuple[0])
N_value=str(KNN_parameters_tuple[2])
P_value='%.2f' % KNN_parameters_tuple[1]
Accuracy='%.2f' % classifier_metrics_tuple[0]
Sensitivity='%.2f' % classifier_metrics_tuple[1]
Specificity='%.2f' % classifier_metrics_tuple[2]

KNN_parameters_tuple=(K_value, P_value, N_value)
classifier_metrics_tuple=(Accuracy, Sensitivity, Specificity)

sys.stdout.write('k: '+K_value+'\n')
sys.stdout.write('p: '+P_value+'\n')
sys.stdout.write('n: '+N_value+'\n')
sys.stdout.write('accuracy: '+Accuracy+'\n')
sys.stdout.write('sensitivity: '+Sensitivity+'\n')
sys.stdout.write('specificity: '+Specificity+'\n')

#creating the output file
output_file=create_output_file(KNN_parameters_tuple, classifier_metrics_tuple)
