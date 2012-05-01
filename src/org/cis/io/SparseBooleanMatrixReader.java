package org.cis.io;

import org.cis.matrix.SparseBooleanMatrix;

/**
 * This interface defining the function to read data of
 * sparse boolean matrix from text file
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public interface SparseBooleanMatrixReader {
	
	/**
	 * Read boolean data from text file
	 * 
	 * @param filePath the path of the text file
	 * @param dim1 the first dimension of the matrix
	 * @param dim2 the second dimension of the matrix
	 * @return
	 */
	public SparseBooleanMatrix read(String filePath, int dim1, int dim2);
}
