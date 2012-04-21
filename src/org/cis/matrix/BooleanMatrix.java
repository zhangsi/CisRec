package org.cis.matrix;

import java.util.ArrayList;

public interface BooleanMatrix {
	
	ArrayList<Integer> getRow( int r);
	
	ArrayList<Integer> getColumn( int c);
	
	int getNonEmpty();
	
	ArrayList<Integer> getNonEmptyRowIds();
	
	ArrayList<Integer> getNonEmptyColumnIds();
	
	int getNumEntriesByRow( int r);
	
	int getNumEntriesByColumn( int c);
}
