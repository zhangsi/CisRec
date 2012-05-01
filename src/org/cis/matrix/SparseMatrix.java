package org.cis.matrix;

import java.util.ArrayList;

/**
 * This interface define the functions of a sparse matrix
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public interface SparseMatrix {
	
	/**
	 * Get the r-th row of the matrix
	 * 
	 * @param the row number
	 * @return the r-th row of the matrix
	 */
	ArrayList<Integer> getRow( int r);
	
	/**
	 * Get the c-th column of the matrix
	 * 
	 * @param the column number
	 * @return the c-th column of the matrix
	 */
	ArrayList<Integer> getColumn( int c);
	
	/**
	 * Get the non empty elements in the matrix
	 * 
	 * @return the number of non empty elements
	 */
	int getNonEmpty();
	
	/**
	 * Get the row-ids of non empty
	 * 
	 * @return the row-ids of non empty
	 */
	ArrayList<Integer> getNonEmptyRowIds();
	
	/**
	 * Get the column-ids of non empty
	 * 
	 * @return the column-ids of non empty
	 */
	ArrayList<Integer> getNonEmptyColumnIds();
	
	/**
	 * Get the number of elements in the given row
	 * 
	 * @param r The row number
	 * @return The number of elements in the r-th row
	 */
	int getNumEntriesByRow( int r);
	
	/**
	 * Get the number of element in the given column
	 * 
	 * @param c The column number
	 * @return The number of elemnts in the c-th column
	 */
	int getNumEntriesByColumn( int c);
}
