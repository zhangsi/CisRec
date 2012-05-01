package org.cis.matrix;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

/**
 * This class implementing a sparse boolean matrix
 * 
 * @author Zhang Si (zhangsi.cs@gmail.com)
 *
 */
public class SparseBooleanMatrix implements SparseMatrix{

	/** the container of store data */
	ArrayList<HashSet<Integer>> row_list;
	
	/** the max row number */
	int maxRow;
	/** the max column number */
	int maxColumn;
	/** the number of non empty elements */
	int count;
	
	/** the first dimension of the matrix */
	int dim1;
	/** the second dimension of the matrix */
	int dim2;
	
	/**
	 * Construct a empty sparse boolean matrix
	 * 
	 * @param the first dimension of the matrix
	 * @param the second dimension of the matrix
	 */
	public SparseBooleanMatrix(int dim1, int dim2) {
		this.row_list = new ArrayList<HashSet<Integer>>();
		this.count = 0;
		this.maxColumn = Integer.MIN_VALUE;
		this.maxRow    = Integer.MIN_VALUE;
		this.dim1 = dim1;
		this.dim2 = dim2;
		
		for(int d = 0; d != dim1; ++d){
			row_list.add(new HashSet<Integer>());
		}

	}
	
	/**
	 * Set the row number of the matrix
	 * @param row
	 */
	public void setRow( int row){
		this.maxRow = row;
	}
	
	/**
	 * Get the row number of the matrix
	 * @return
	 */
	public int getRow(){
		return this.maxRow;
	}
	
	/**
	 * Set the column number of the matrix
	 * @param column
	 */
	public void setColumn( int column) {
		this.maxColumn = column;
	}
	
	/**
	 * Get the column number of the matrix
	 * @return
	 */
	public int getColumn() {
		return this.maxColumn;
	}
	
	/**
	 * Get an element in the sparse matrix
	 * 
	 * @param r the row index of the element
	 * @param c the column index of the element
	 * @return the r-row c-column element of the matrix
	 */
	public boolean getEntry( int r, int c){
		if( r > maxRow || c > maxColumn)
			return false;
		return row_list.get(r).contains(c);
	}
	
	/**
	 * Set an element to true in the sparse matrix
	 * 
	 * @param r the row index of the element
	 * @param c the column index of the element
	 */
	public void setEntry( int r, int c){
		if( r > maxRow || c > maxColumn)
			return;
		row_list.get(r).add(c);
	}
	
	/**
	 * Add an element to sparse matrix
	 * 
	 * @param r the row index of the element
	 * @param c the column index of the element
	 */
	public void addEntry(int r ,int c) {

		row_list.get(r).add(c);
		
		count++;
		
		if( r > maxRow)
			maxRow = r;
		if( c > maxColumn)
			maxColumn = c;
	}
	
	/**
	 * Transpose of the matrix: A -> A'
	 * 
	 * @return the transpose of the sparse matrix
	 */
	public SparseBooleanMatrix transpose() {
		SparseBooleanMatrix matrix = new SparseBooleanMatrix(dim2, dim1);
		
		matrix.count = this.count;
		matrix.setColumn(this.maxColumn);
		matrix.setRow(this.maxRow);
		
		for( int row_id = 0; row_id <= maxRow; ++row_id){
			Iterator<Integer> iterator = row_list.get(row_id).iterator();
			while(iterator.hasNext()){
				matrix.setEntry(iterator.next(), row_id);
			}
		}
		
		return matrix;
	}
	
	/**
	 * Get the r-th row of the matrix
	 * 
	 * @param the row number
	 * @return the r-th row of the matrix
	 */
	public ArrayList<Integer> getRow(int r) {
		if( r > dim1)
			return null;
		return new ArrayList<Integer>(row_list.get(r));
	}

	/**
	 * Get the c-th column of the matrix
	 * 
	 * @param the column number
	 * @return the c-th column of the matrix
	 */
	public ArrayList<Integer> getColumn(int c) {
		if( c > maxColumn)
			return null;
		ArrayList<Integer> list = new ArrayList<Integer> ();
		for( int row_id = 0; row_id <= maxRow; ++row_id){
			if(row_list.get(row_id).contains(c)){
				list.add(row_id);
			}
		}
		return list;
	}

	/**
	 * Get the non empty elements in the matrix
	 * 
	 * @return the number of non empty elements
	 */
	public int getNonEmpty() {
		return this.count;
	}

	/**
	 * Get the row-ids of non empty
	 * 
	 * @return the row-ids of non empty
	 */
	public ArrayList<Integer> getNonEmptyRowIds() {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for(int row_id = 0; row_id <= maxRow; ++row_id){
			if(!row_list.get(row_id).isEmpty())
				list.add(row_id);
		}
		return list;
	}

	/**
	 * Get the column-ids of non empty
	 * 
	 * @return the column-ids of non empty
	 */
	public ArrayList<Integer> getNonEmptyColumnIds() {
		HashSet<Integer> set = new HashSet<Integer>();
		for(int row_id = 0; row_id != maxRow; ++row_id){
			Iterator<Integer> iterator = row_list.get(row_id).iterator();
			while(iterator.hasNext()){
				set.add(iterator.next());
			}
		}
		return new ArrayList<Integer>(set);
	}

	/**
	 * Get the number of elements in the given row
	 * 
	 * @param r The row number
	 * @return The number of elements in the r-th row
	 */
	public int getNumEntriesByRow(int r) {
		if( r > maxRow)
			return -1;
		return row_list.get(r).size();
	}

	/**
	 * Get the number of element in the given column
	 * 
	 * @param c The column number
	 * @return The number of elemnts in the c-th column
	 */
	public int getNumEntriesByColumn(int c) {
		if( c > maxColumn)
			return -1;
		int num = 0;
		for(int row_id = 0; row_id <= maxRow; ++row_id){
			if(row_list.get(row_id).contains(c))
				num++;
		}
		return num;
	}

}
