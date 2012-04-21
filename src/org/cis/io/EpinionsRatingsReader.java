package org.cis.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.cis.data.Ratings;

public class EpinionsRatingsReader implements RatingsReader{

	@Override
	public Ratings read(String filePath) {
		Ratings ratings = new Ratings();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(filePath));
			String line;
			String[] words;
			int u,i,r;
			
			int count = 0;
			while( (line = br.readLine()) != null){
				words = line.split("\t");
				u = Integer.parseInt(words[0]);
				i = Integer.parseInt(words[1]);
				r = Integer.parseInt(words[2]);
				ratings.addRating(u, i, r);
				count++;
			}
			System.out.println("read file: " + filePath + " end. The total line number is: " + count);
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ratings;
	}

}
