#! /home/deeto/anaconda3/bin/python

# Maria Ahmad
# BMED 8813
# HW 6: Problem 2
# Needleman-Wunsch Algorithm
# go bioinformatics!





###############################################################################
###############################################################################
###############################################################################
import sys
###############################################################################
###############################################################################
###############################################################################





###############################################################################
###############################################################################
###############################################################################
def parse_scoring_matrix(csvFile):
    # open scoring matrice file
    handle = open(csvFile,'r')
    # the first line contains the gap penalty and the base pairs
    first_line = handle.readline()
    header_line = first_line.strip().split(',')
    gap_penalty = int(header_line[0])
    # initialize scoring matrix dictionary
    scoring_matrix = {}
    # the other lines contain the match / mismatch scores
    otherLines = handle.readlines()
    # initialize the index count (start at 1 because index 0 is for gap penalty
    count = 1
    # for A,G,C,T
    for basepair in header_line[1:]:
        # For every base pair, iterate through the lines and index the score that matches
        for other_line in otherLines:
            other_line = other_line.strip().split(',')
            basepair2 = other_line[0]
            score = other_line[count]
            # add to the dictionary
            scoring_matrix[basepair+':'+basepair2] = int(score)
        # iterate through the bps
        count += 1
    return gap_penalty, scoring_matrix
###############################################################################
###############################################################################
###############################################################################





###############################################################################
###############################################################################
###############################################################################
def initialize_matrix(sequence1,sequence2,gap_penalty,scoring_matrix):
    length1 = len(sequence1)
    length2 = len(sequence2)
    
    # initialize the scores of the first row and column
    column_i = list(range(0,gap_penalty*(length1+1),gap_penalty))
    row_i = list(range(gap_penalty,gap_penalty*(length2+1),gap_penalty))
    
    return column_i, row_i
###############################################################################
###############################################################################
###############################################################################





###############################################################################
###############################################################################
###############################################################################
def fill_matrix(sequence1,sequence2,gap_penalty,scoring_matrix,column_i, row_i):
    # initialize all the rows in the matrix
    matrix_rows = [row_i]
    
    # initialize counts
    column_count = 0
    seq1_count = 1
    letter1_count = 1
    
    # initialize matrices
    initial_matrix = {}
    
    # backtracing dictionary
    backtracing = {}
    
    # iterate through the first sequence
    for letter1 in sequence1: # sequence1 is going horizontal, so iterating through the columns
        letter2_count = 1 
        
        # initialize the row counts, sequence2 is going vertical, so iterating through the rows
        row_count = 0
        # initialize the new row list
        row = []
        
        # iterate through the second sequence
        for letter2 in sequence2:
            if letter2_count == 1: # this means that it is the first letter in the row
                diag = column_i[column_count] # diagonal
                left = column_i[column_count + 1] 
            else: # the letter is in the middle/end of the row
                # square has already been created in the steps below
                diag = square[1]
                left = square[3]
            
            top_right = matrix_rows[column_count][row_count]
            
            # key to find corresponding match/mismatch score
            key = letter1 + ':' + letter2
            match_mismatch_score = scoring_matrix[key]
            
            # hypothetical corners in the square: diagonal, left, top right
            hyp_diag = diag + match_mismatch_score # hypothetical diagonal
            hyp_left = left + gap_penalty
            hyp_top_right = top_right + gap_penalty
            
            # max from diagonal, left, top right will become the new bottom right
            bottom_right = max(hyp_diag, hyp_left, hyp_top_right)
            
            # add value to the row
            row.append(bottom_right)
            
            # update the square values
            square = [diag, top_right, left, bottom_right]
            # square2 is for backtracking purposes later on
            square2 = [hyp_diag, hyp_top_right, hyp_left, bottom_right]
            
            # coordinate and coordinate name for the matrix
            coordinate = str(letter1_count) + ':' + str(letter2_count)
            coordinate_name = key
            
            # update the initial matrix
            initial_matrix[coordinate] = [square, square2, coordinate_name]
            
            ### new line!
            # tells you if there are duplicate maximums
            max_counts = [hyp_diag, hyp_left, hyp_top_right].count(bottom_right)
            backtracing[coordinate] = set() 
            if hyp_diag == bottom_right:
                # must parse the coordinates because they were strings
                coordinates = coordinate.split(':')
                coordinate1 = coordinates[0]
                coordinate2 = coordinates[1]
                coordinate1 = int(coordinate1) - 1 # change row count
                coordinate2 = int(coordinate2) - 1 # change column count
                prev_coordinate = str(coordinate1)+':'+str(coordinate2)
                backtracing[coordinate].add(prev_coordinate)
            if hyp_left == bottom_right:
                coordinates = coordinate.split(':')
                coordinate1 = coordinates[0]
                coordinate2 = coordinates[1]
                coordinate2 = int(coordinate2) - 1 # change column count
                prev_coordinate = str(coordinate1)+':'+str(coordinate2)
                backtracing[coordinate].add(prev_coordinate)
            if hyp_top_right == bottom_right:
                coordinates = coordinate.split(':')
                coordinate1 = coordinates[0]
                coordinate2 = coordinates[1]
                coordinate1 = int(coordinate1) - 1 # change row count
                prev_coordinate = str(coordinate1)+':'+str(coordinate2)
                backtracing[coordinate].add(prev_coordinate)
                
            # increment 
            letter2_count += 1
            row_count += 1
            
        # add to the matrix rows
        matrix_rows.append(row)
        
        # iterate
        letter1_count += 1
        column_count += 1
        
    # matrix is now filled
    filled_matrix = initial_matrix
        
    return filled_matrix, backtracing
###############################################################################
###############################################################################
###############################################################################





###############################################################################
###############################################################################
###############################################################################
def backtrack(filled_matrix,backtracing,seq1,seq2):
    # initialize list of matrix coordinates
    matrix_coordinates = list(filled_matrix.keys())
        
    # sort the keys 
    matrix_coordinates.sort()
    
    # final point in the alignment path (the maximum score)
    final_point = str(len(seq1))+':'+str(len(seq2))
    
    # recursive function of tracing multiple paths
    def find_all_paths(start, end, graph, visited=None):
        if visited is None:
            visited = set()

        visited |= {start}
        for node in graph[start]:
            if node == end:
                yield [start,end]
            else:
                for pathx in find_all_paths(node, end, graph, visited):
                    yield [start] + pathx
                    
    all_paths = find_all_paths(final_point, '1:1', backtracing)
    
    # initialize sequence alignment lists
    sequence1_alignments = []
    sequence2_alignments = []
    
    # create the actual alignment
    for patho in all_paths:
        new_path = patho[::-1] # reverses the path so you can read it going forward
        initial_coordinate = new_path[0]

        # initialize the alignments
        seqVseq = filled_matrix[new_path[0]]
        seqVseq = filled_matrix[new_path[0]][2]
        sequence1_alignment = seqVseq[0]
        sequence2_alignment = seqVseq[2]

        # iterate through every coordinate in the path, and get the alignment sequences
        for coordinate in new_path[1:]: # already used the first coordinate to initialize the alignments
            # the sequences involved in the particular coordinate
            basepairs = filled_matrix[coordinate][2]
            
            # parse through the coordinates as before
            coordinates = coordinate.split(':')
            coordinate1 = coordinates[0]
            coordinate2 = coordinates[1]
            
            # parse through the initial coordinates as well
            initial_coordinates = initial_coordinate.split(':')
            ic1 = initial_coordinates[0]
            ic2 = initial_coordinates[1]
            
            if (int(coordinate1) - int(ic1)) == 1: # belongs one row above
                if (int(coordinate2) - int(ic2)) == 1: # belongs one column to the side, therefore a diagonal piece
                    sequence1_alignment += basepairs[0]
                    sequence2_alignment += basepairs[2]
                else: # just a piece directly above
                    sequence1_alignment += basepairs[0]
                    sequence2_alignment += '-'
            else: # a piece to the left
                sequence1_alignment += '-'
                sequence2_alignment += basepairs[2]
            initial_coordinate = coordinate
        
        # add to the sequence alignment lists
        sequence1_alignments.append(sequence1_alignment)
        sequence2_alignments.append(sequence2_alignment)

    
    # print the alignments
    string = ""
    for alignment in sequence1_alignments:
        string += alignment + '\t'
    print(string)
    string = ""
    for alignment in sequence2_alignments:
        string += alignment + '\t'
    print(string)


    return
###############################################################################
###############################################################################
###############################################################################







###############################################################################
###############################################################################
###############################################################################
def main():
    sequence1 = sys.argv[1]
    sequence2 = sys.argv[2]
    csvFile = sys.argv[3]
    
    # parse the scoring matrix for match, mismatch, and gap penalties
    gap_penalty, scoring_matrix = parse_scoring_matrix(csvFile)
    
    # initialize the matrix
    column_i, row_i = initialize_matrix(sequence1, sequence2, gap_penalty, scoring_matrix)
    
    # fill the matrix
    filled_matrix, backtracing = fill_matrix(sequence1,sequence2,gap_penalty,scoring_matrix,column_i, row_i)
    
    # backtrack through the matrix to find the alignment path
    backtrack(filled_matrix, backtracing,sequence1,sequence2)
###############################################################################
###############################################################################
###############################################################################  
#     def function(the_point,history):
#         if the_point == 11:
# #             history.append(the_point)
#             print(history)
#             paths.append(history)
# #             bbpath = [final_point]
#             return 
#         else:
#             previous_points = backtracing[the_point]
#             for point in previous_points:
#                 history.append(point)
#                 function(point,history)

#     def function(point):
#         if point == 11:
#             return point
#         else:
#             previous_points = backtracking[point]
#             for point in previous_points
#                 return function(point)
#     previous_points = backtracing[final_point]
#     for point in previous_points:
#         bbpath.append(point)
#         function(point)
#     path_list = [path]
    
#     def function(path,point):
#         back_to_square_one = filled_matrix[point]
#         diagonal = back_to_square_one[1][0]
#         top_right = back_to_square_one[1][1]
#         left = back_to_square_one[1][2]    
    
main()
'''
while final_point != 11: # 11 is the coordinate for the final point in the path, which we are trying to get to
        back_to_square_one = filled_matrix[final_point]
        diagonal = back_to_square_one[1][0]
        top_right = back_to_square_one[1][1]
        left = back_to_square_one[1][2]
        
        # when backtracing, you go to the square in which the query square arose from 
        if diagonal == max(diagonal, top_right, left):
            final_point = final_point - 11 # diagonal square is 11 coordinates away 
            path.append(final_point)
        elif top_right == max(diagonal, top_right, left):
            final_point = final_point - 10 # directly above square is 10 coordinates away
            path.append(final_point)
        elif left == max(diagonal, top_right, left):
            final_point = final_point - 1 # left square is 1 coordinate away
            path.append(final_point)
    print(final_point)
'''

#     new_path = path[::-1] # reverses the path so you can read it going forward
#     initial_coordinate = new_path[0]
    
#     # initialize the alignments
#     seqVseq = filled_matrix[new_path[0]][2]
#     sequence1_alignment = seqVseq[0]
#     sequence2_alignment = seqVseq[2]
    
#     # iterate through every coordinate in the path, and get the alignment sequences
#     for coordinate in new_path[1:]: # already used the first coordinate to initialize the alignments
#         basepairs = filled_matrix[coordinate][2]
#         difference = coordinate - initial_coordinate
#         if difference == 11: # diagonal
#             sequence1_alignment += basepairs[0]
#             sequence2_alignment += basepairs[2]
#         elif difference == 10: # directly above
#             sequence1_alignment += basepairs[0]
#             sequence2_alignment += '-'
#         elif difference == 1: # directly to the side
#             sequence1_alignment += '-'
#             sequence2_alignment += basepairs[2]
#         initial_coordinate = coordinate
#     print('\nAlignments:')
#     print(sequence1_alignment)
#     print(sequence2_alignment)
#             difference = coordinate - initial_coordinate
#             if difference == 11: # diagonal
#                 sequence1_alignment += basepairs[0]
#                 sequence2_alignment += basepairs[2]
#             elif difference == 10: # directly above
#                 sequence1_alignment += basepairs[0]
#                 sequence2_alignment += '-'
#             elif difference == 1: # directly to the side
#                 sequence1_alignment += '-'
#                 sequence2_alignment += basepairs[2]
#             initial_coordinate = coordinate