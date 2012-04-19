package org.cis.io;

import org.cis.data.Ratings;

public interface RatingsReader {
	
	public Ratings read(String filePath);

}
