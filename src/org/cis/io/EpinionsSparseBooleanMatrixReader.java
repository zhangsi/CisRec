package org.cis.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.cis.matrix.SparseBooleanMatrix;

public class EpinionsSparseBooleanMatrixReader implements
		SparseBooleanMatrixReader {

	@Override
	public SparseBooleanMatrix read(String filePath) {
		SparseBooleanMatrix matrix = new SparseBooleanMatrix();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(filePath));
			String line;
			String[] words;
			int u,i;
			
			int count = 0;
			while( (line = br.readLine()) != null){
				words = line.split("\t");
				u = Integer.parseInt(words[0]);
				i = Integer.parseInt(words[1]);
				matrix.addEntry(u, i);
				count++;
			}
			System.out.println("read file: " + filePath + " end. The total line number is: " + count);
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return matrix;
	}

}
