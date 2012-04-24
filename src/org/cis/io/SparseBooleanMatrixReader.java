package org.cis.io;

import org.cis.matrix.SparseBooleanMatrix;

public interface SparseBooleanMatrixReader {
	
	public SparseBooleanMatrix read(String filePath, int dim1, int dim2);
}
