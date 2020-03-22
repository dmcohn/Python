# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Wed Oct 21 18:16:48 2015

@author: David
"""
import sys
import numpy
import random

"""
This function takes in a list of lines from an expression data input file,
partitions the data based on their proper row and column, and inserts the data
into an appropriately sized matrix.

Input:
    expression_file_lines: list of raw lines, with their tags still included, 
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
This function randomly generates initial centroids to use for the kmeans algorithm.
This function is only called if the user does not provide an input file with 
pre-determined centroids to use. Each element within randomly generated centroids are contained
within the range of values encompassed by the data set for that element. 

Inputs:
    expression_matrix: a matrix that contains the expression data across different
    genes and conditions; in this matrix, rows refer to genes or other measurements,
    and columns refer to experiments.
    k: number of clusters for the algorithm
Returns: A matrix that contains the k initial centroids, which each centroid getting
its own row.
"""    
def generate_initial_centroids(expression_matrix, k):
    centroids_list=[]
    centroids_matrix=numpy.zeros((k,expression_matrix.shape[1]))
    for i in xrange(0,k):
        proposed_centroid=[]
        centroid_boolean=False
        centroid_string_representation=""
        while(centroid_boolean!=True):
            for j in xrange(0, expression_matrix.shape[1]):
                column_vector=expression_matrix[:,j]
                numpy.sort(column_vector)
                maximum=column_vector[len(column_vector)-1]
                proposed_centroid_value=random.uniform(column_vector[0]+0.0001,maximum)
                proposed_centroid.extend([proposed_centroid_value])
                centroid_string_representation=centroid_string_representation+" "+str(proposed_centroid[j])
            if(centroid_string_representation not in centroids_list):
                centroids_list.extend([centroid_string_representation])
                for j in xrange(0, len(proposed_centroid)):
                    centroids_matrix[i,j]=proposed_centroid[j]
                centroid_boolean=True
    return centroids_matrix
"""
The function updates the centroids for each cluster based on average of element values
across all vectors in the cluster. 
Inputs:
    expression_matrix: a matrix that contains the expression data across different
    genes and conditions; in this matrix, rows refer to genes or other measurements,
    and columns refer to experiments.
    centroid_matrix: matrix containing the centroids for the previous iteration
    cluster_to_gene_assignment_dictionary: dictionary, that stores for each cluster,
    a list of genes represented by their corresponding indices
Returns: a matrix containing the new k centroids, based on the average of element values
across all vectors in that clusters; each of the k centroids gets its own row in this matrix.
"""
def update_centroids(expression_matrix, centroid_matrix, cluster_to_gene_assignment_dictionary):
    updated_centroid_matrix=numpy.zeros((centroid_matrix.shape[0],expression_matrix.shape[1]))
    for i in xrange(0,centroid_matrix.shape[0]):
        if(cluster_to_gene_assignment_dictionary.has_key(str(i+1))):
            cluster_member_list=cluster_to_gene_assignment_dictionary[str(i+1)]
            cluster_member_matrix=numpy.zeros((len(cluster_member_list),expression_matrix.shape[1]))
            for l in xrange(0, len(cluster_member_list)):
                index=cluster_member_list[l]
                cluster_member_matrix[l,:]=expression_matrix[index,:]
            for j in xrange(0, expression_matrix.shape[1]):
                updated_centroid_matrix[i,j]=numpy.average(cluster_member_matrix[:,j])
    return updated_centroid_matrix
"""
This function assigns each gene in the expression matrix to its proper cluster, by determining
the minimum Euclidean distance between the vector representing the gene, and the centroid representing the cluster.
This function, after determining the proper cluster based on Euclidean distance, will update the dictionary that
map every gene to a cluster number given its index, as well as a dictionary that maps a cluster number to a list of indices
representing each of the genes in that cluster. 

Inputs:
    expression_matrix: a matrix that contains the expression data across different
    genes and conditions; in this matrix, rows refer to genes or other measurements,
    and columns refer to experiments.
    centroid_matrix: matrix containing the centroids for each of the k clusters; in this matrix,
    each row corresponds to a different centroid
    maximum_iterations: the maximum number of iterations of cluster assignment and centroid updating
    that the kmeans algorithm will perform, if the algorithm does not converge prior to this maximum
    number of iterations being reached
    Number_of_iterations=number of iterations of kmeans that are run, either prior to the algorithm
    converging around a particular set of k clusters, or until the maximum number of iterations is reached
Returns: k_means_cluster_tuple, which contains the dictionary that maps every gene to a cluster number given
its index. In addition, k_means_cluster_tuple also contains the number of iterations needed for the algorithm
to converge around a set of k clusters, or the maximum number of allowed iterations if the algorithm did not converge
fast enough.
"""        
def assign_clusters(expression_matrix, centroid_matrix, maximum_iterations, Number_of_iterations):
    cluster_to_gene_assignment_dictionary={}
    gene_to_cluster_assignment_dictionary={}
    for k in xrange(0, maximum_iterations):
        Number_of_cluster_changes=0
        for i in xrange(0, expression_matrix.shape[0]):
            gene_vector=expression_matrix[i,:]
            cluster_number=0
            for j in xrange(0, centroid_matrix.shape[0]):
                centroid=centroid_matrix[j,:]
                gene_centroid_distance=numpy.linalg.norm(gene_vector-centroid)
                if(j==0):
                    minimum_distance=gene_centroid_distance
                    cluster_number=j+1
                else:
                    if(gene_centroid_distance<minimum_distance):
                        minimum_distance=gene_centroid_distance
                        cluster_number=j+1
            # updating gene-to-cluster, and cluster-to-gene dictionaries            
            if(gene_to_cluster_assignment_dictionary.has_key(str(i+1))):
                previous_cluster_assignment=gene_to_cluster_assignment_dictionary[str(i+1)]
                if(previous_cluster_assignment!=cluster_number):
                    Number_of_cluster_changes+=1
                    gene_to_cluster_assignment_dictionary[str(i+1)]=cluster_number
                    previous_cluster_list=cluster_to_gene_assignment_dictionary[str(previous_cluster_assignment)]
                    previous_cluster_list.remove(i)
                    cluster_to_gene_assignment_dictionary[str(previous_cluster_assignment)]=previous_cluster_list
            else:
                gene_to_cluster_assignment_dictionary[str(i+1)]=cluster_number
                Number_of_cluster_changes+=1
                
            if(cluster_to_gene_assignment_dictionary.has_key(str(cluster_number))):
                cluster_member_list=cluster_to_gene_assignment_dictionary[str(cluster_number)]
                if(i not in cluster_member_list):
                    cluster_member_list.extend([i])
                    cluster_to_gene_assignment_dictionary[str(cluster_number)]=cluster_member_list 
            else:
                cluster_to_gene_assignment_dictionary[str(cluster_number)]=[i]
        centroid_matrix=update_centroids(expression_matrix, centroid_matrix, cluster_to_gene_assignment_dictionary)
        if(Number_of_cluster_changes==0):
            Number_of_iterations=k
            break
        Number_of_iterations=k+1
    k_means_cluster_tuple=(gene_to_cluster_assignment_dictionary, Number_of_iterations)
    return k_means_cluster_tuple
"""
This function takes a dictionary that maps every gene to a cluster number given its index,
and stores this information in a more readily accessible matrix. In this matrix,
each gene has its own row, with the gene number being stored in the first column, and the
cluster number being stored in the second column.

Inputs:
    expression_matrix:a matrix that contains the expression data across different
    genes and conditions; in this matrix, rows refer to genes or other measurements,
    and columns refer to experiments.
    final_clusters_tuple: a tuple which contains the dictionary that maps every gene to a cluster number given
its index. In addition, this tuple also contains the number of iterations needed for the algorithm
to converge around a set of k clusters, or the maximum number of allowed iterations if the algorithm did not converge
fast enough. 
"""
def generate_final_clusters_matrix(expression_matrix, final_clusters_tuple):
    final_clusters_matrix=numpy.zeros((expression_matrix.shape[0],2))
    gene_list=(final_clusters_tuple[0]).keys()
    for i in xrange(0,len(gene_list)):
        gene_number=int(gene_list[i])
        final_clusters_matrix[gene_number-1,0]=gene_number
        final_clusters_matrix[gene_number-1,1]=(final_clusters_tuple[0])[gene_list[i]]    
    return final_clusters_matrix
    
k=int(sys.argv[1])
expression_file=open(sys.argv[2])
maximum_iterations=int(sys.argv[3])
expression_file_lines=expression_file.readlines()
#Reading inputted expression data into matrix
expression_matrix=derive_expression_matrix(expression_file_lines)
#generating centroids based on user inputted centroid files
if(len(sys.argv)==5):
    centroid_file=open(sys.argv[4])
    centroid_file_lines=centroid_file.readlines()
    centroid_matrix=derive_expression_matrix(centroid_file_lines)
    if(centroid_matrix.shape[0]>k):
        centroid_matrix=centroid_matrix[0:k,:]
else:
    #Randomly generates centroids to use
    centroid_matrix=generate_initial_centroids(expression_matrix, k)
    
Number_of_iterations=0
#Kmeans cluster assignment algorithm
final_clusters_tuple=assign_clusters(expression_matrix, centroid_matrix, maximum_iterations, Number_of_iterations)

final_clusters_matrix=generate_final_clusters_matrix(expression_matrix, final_clusters_tuple)

#Outputting Kmeans results to output file
output_file=open('kmeans.out', 'w')  
for i in xrange(0, final_clusters_matrix.shape[0]):
    gene_number=int(final_clusters_matrix[i,0])
    cluster_number=int(final_clusters_matrix[i,1])
    output_file.write(''+str(gene_number)+'\t'+str(cluster_number)+'\n')
output_file.close()

sys.stdout.write('iterations: '+str(int(final_clusters_tuple[1]))+'\n')
