package org.cis.matrix;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

public class SparseBooleanMatrix implements BooleanMatrix{

	ArrayList<HashSet<Integer>> row_list;
	
	int maxRow;
	int maxColumn;
	int count;
	
	int dim1;
	int dim2;
	
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
	
	public void setRow( int row){
		this.maxRow = row;
	}
	
	public int getRow(){
		return this.maxRow;
	}
	
	public void setColumn( int column) {
		this.maxColumn = column;
	}
	
	public int getColumn() {
		return this.maxColumn;
	}
	
	public boolean getEntry( int r, int c){
		if( r > maxRow || c > maxColumn)
			return false;
		return row_list.get(r).contains(c);
	}
	
	public void setEntry( int r, int c){
		if( r > maxRow || c > maxColumn)
			return;
		row_list.get(r).add(c);
	}
	
	public void addEntry(int r ,int c) {

		row_list.get(r).add(c);
		
		count++;
		
		if( r > maxRow)
			maxRow = r;
		if( c > maxColumn)
			maxColumn = c;
	}
	
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
	
	@Override
	public ArrayList<Integer> getRow(int r) {
		if( r > dim1)
			return null;
		return new ArrayList<Integer>(row_list.get(r));
	}

	@Override
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

	@Override
	public int getNonEmpty() {
		return this.count;
	}

	@Override
	public ArrayList<Integer> getNonEmptyRowIds() {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for(int row_id = 0; row_id <= maxRow; ++row_id){
			if(!row_list.get(row_id).isEmpty())
				list.add(row_id);
		}
		return list;
	}

	@Override
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

	@Override
	public int getNumEntriesByRow(int r) {
		if( r > maxRow)
			return -1;
		return row_list.get(r).size();
	}

	@Override
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
