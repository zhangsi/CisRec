package org.cis.util;

/**
 * this class sets vector of matrix to zero
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class ZeroSetter {
	
	/**
	 * set all the elements of a double vector to zero
	 * 
	 * @param the double vector to be set to zero
	 * @param the length of the vector
	 */
	public static void zero(double[] a, int l1) {
		for(int i = 0; i < l1; i++)
				a[i] = 0;
	}
	
	/**
	 * set all the elements of a integer vector to zero
	 * 
	 * @param the integer vector to be set to zero
	 * @param the length of the vector
	 */
	public static void zero(int[] a, int l1) {
		for(int i = 0; i < l1; i++)
				a[i] = 0;
	}
	
	/**
	 * set all the elements of a integer matrix to zero
	 * 
	 * @param the integer matrix to be set to zero
	 * @param the rows of the matrix
	 * @param the columns of the matrix
	 */
	public static void zero(int[][] a, int l1, int l2) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				a[i][j] = 0;
	}
	
	/**
	 * set all the elements of a double matrix to zero
	 * 
	 * @param the double matrix to be set to zero
	 * @param the rows of the matrix
	 * @param the columns of the matrix
	 */
	public static void zero(double[][] a, int l1, int l2) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				a[i][j] = 0.0;
	}
	
	/**
	 * set all the elements of a double tensor to zero
	 * 
	 * @param the double tensor to be set to zero
	 * @param the first dimension of the tensor
	 * @param the second dimension of the tensor
	 * @param the third dimension of the tensor
	 */
	public static void zero(double[][][] a, int l1, int l2, int l3) {
		for(int i = 0; i < l1; i++)
			for(int j = 0; j < l2; j++)
				for(int k = 0; k < l3; k++)
					a[i][j][k] = 0.0;
	}
}
